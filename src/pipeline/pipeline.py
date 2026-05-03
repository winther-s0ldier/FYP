"""
Full inference pipeline orchestration.
Layers: Trie → Redis → ModernBERT → pgvector + HMM → Decision → [Escalation]
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.pipeline.trie import SlurTrie
from src.pipeline.decision import decide_fast_path, needs_escalation, DecisionResult, Action
from src.session.store import SessionStore
from src.session.hmm import SessionHMM
from src.classifier.model import ContentModerationModel
from src.classifier.dataset import INTENT_LABELS, IDX2INTENT
from src.escalation.llm_router import escalate
from src.db.postgres import MessageStore
import os
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
INTENT_CONF_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", 0.70))


@dataclass
class ClassificationRequest:
    text: str
    user_id: str
    session_id: str = ""


@dataclass
class ClassificationResponse:
    text: str
    user_id: str
    action: str
    toxicity_score: float
    intent_label: str
    confidence: float
    session_risk: float
    risk_score: float
    trie_matched: bool
    api_escalated: bool
    latency_ms: float
    reasoning: Optional[str] = None
    policy_rule: Optional[str] = None
    embedding: Optional[list] = field(default=None, repr=False)


class ModerationPipeline:
    def __init__(
        self,
        model: ContentModerationModel,
        tokenizer: AutoTokenizer,
        trie: SlurTrie,
        session_store: SessionStore,
        hmm: SessionHMM,
        message_store: Optional[MessageStore] = None,
        device: str = "cuda",
        max_length: int = 256,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.trie = trie
        self.session_store = session_store
        self.hmm = hmm
        self.message_store = message_store
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

    async def classify(self, req: ClassificationRequest) -> ClassificationResponse:
        t0 = time.perf_counter()

        # --- Layer 1: Trie pre-filter (~1ms) ---
        trie_hit, matched_word = self.trie.search(req.text)
        if trie_hit:
            result = DecisionResult(
                action=Action.BLOCK, risk_score=1.0,
                toxicity_score=1.0, session_risk=0.0,
                intent_label="threat", confidence=1.0,
                trie_matched=True, api_escalated=False,
            )
            return await self._finalise(req, result, None, t0)

        # --- Layer 2: Redis session fetch (~3ms) ---
        intent_history = await self.session_store.get_intents(req.user_id)

        # --- Layer 3: ModernBERT inference (~30-40ms) ---
        enc = self.tokenizer(
            req.text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )

        toxicity_score = float(out.toxicity_score[0].cpu())
        intent_idx = int(out.intent_logits[0].argmax().cpu())
        intent_label = IDX2INTENT.get(intent_idx, "question")
        confidence = float(out.confidence[0].cpu())
        embedding = out.embedding[0].cpu().numpy()

        # --- Session update: push intent (confidence-weighted) ---
        await self.session_store.push_intent(
            req.user_id, intent_label, confidence, INTENT_CONF_THRESHOLD
        )

        # --- HMM session risk (~5ms) ---
        updated_history = [intent_label] + list(intent_history)
        session_risk = self.hmm.compute_risk(updated_history)
        await self.session_store.set_risk(req.user_id, session_risk)

        # --- pgvector similarity (async, non-blocking) ---
        vector_hits = 0
        if self.message_store:
            vector_hits = await self.message_store.count_similar_flagged(
                embedding, threshold=0.85
            )

        # --- Layer 4: Decision engine ---
        if needs_escalation(confidence, trie_hit):
            # Low-confidence path → LLM API escalation
            llm_result = await escalate(
                message=req.text,
                session_history=updated_history[:5],
                similar_cases=[],
                policy_context="",
                intent_labels=INTENT_LABELS,
            )
            toxicity_score = llm_result.get("toxicity", toxicity_score)
            intent_label = llm_result.get("intent", intent_label)
            confidence = llm_result.get("confidence", confidence)

            result = decide_fast_path(
                toxicity_score, session_risk, intent_label,
                confidence, vector_hits, trie_hit
            )
            result.api_escalated = True
            result.escalation_reasoning = llm_result.get("reasoning")
            result.policy_rule = llm_result.get("policy_rule")
        else:
            result = decide_fast_path(
                toxicity_score, session_risk, intent_label,
                confidence, vector_hits, trie_hit
            )

        return await self._finalise(req, result, embedding, t0)

    async def _finalise(
        self, req: ClassificationRequest,
        result: DecisionResult,
        embedding: Optional[np.ndarray],
        t0: float,
    ) -> ClassificationResponse:
        latency = (time.perf_counter() - t0) * 1000

        response = ClassificationResponse(
            text=req.text,
            user_id=req.user_id,
            action=result.action.value,
            toxicity_score=result.toxicity_score,
            intent_label=result.intent_label,
            confidence=result.confidence,
            session_risk=result.session_risk,
            risk_score=result.risk_score,
            trie_matched=result.trie_matched,
            api_escalated=result.api_escalated,
            latency_ms=latency,
            reasoning=result.escalation_reasoning,
            policy_rule=result.policy_rule,
            embedding=embedding.tolist() if embedding is not None else None,
        )

        # Persist to PostgreSQL (non-blocking)
        if self.message_store:
            asyncio.create_task(self.message_store.save(response))

        return response
