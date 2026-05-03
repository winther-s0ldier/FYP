"""
Decision engine — fuse toxicity score, session risk, vector hits, confidence.
Routes to fast-path decision or LLM escalation.
"""
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", 0.85))
FLAG_THRESHOLD = float(os.getenv("FLAG_THRESHOLD", 0.65))
WARN_THRESHOLD = float(os.getenv("WARN_THRESHOLD", 0.40))

# Fusion weights (tuned empirically; ablation shows these outperform equal weighting)
W_TOXICITY = 0.50
W_SESSION = 0.35
W_VECTOR = 0.15


class Action(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    FLAG = "flag"
    BLOCK = "block"
    ESCALATE = "escalate"  # internal — resolved to one of above after LLM call


@dataclass
class DecisionResult:
    action: Action
    risk_score: float
    toxicity_score: float
    session_risk: float
    intent_label: str
    confidence: float
    trie_matched: bool
    api_escalated: bool
    escalation_reasoning: str | None = None
    policy_rule: str | None = None


def fuse_risk(
    toxicity_score: float,
    session_risk: float,
    vector_hits: int,
) -> float:
    """Weighted fusion of three risk signals."""
    vector_signal = 1.0 if vector_hits > 0 else 0.0
    return (
        W_TOXICITY * toxicity_score
        + W_SESSION * session_risk
        + W_VECTOR * vector_signal
    )


def decide_fast_path(
    toxicity_score: float,
    session_risk: float,
    intent_label: str,
    confidence: float,
    vector_hits: int,
    trie_matched: bool,
) -> DecisionResult:
    """
    Fast-path decision (no LLM). Used when confidence >= CONFIDENCE_THRESHOLD
    or when trie matched (instant block).
    """
    # Trie match → instant block regardless of confidence
    if trie_matched:
        return DecisionResult(
            action=Action.BLOCK,
            risk_score=1.0,
            toxicity_score=toxicity_score,
            session_risk=session_risk,
            intent_label=intent_label,
            confidence=1.0,
            trie_matched=True,
            api_escalated=False,
        )

    risk = fuse_risk(toxicity_score, session_risk, vector_hits)

    if risk > BLOCK_THRESHOLD:
        action = Action.BLOCK
    elif risk > FLAG_THRESHOLD:
        action = Action.FLAG
    elif risk > WARN_THRESHOLD:
        action = Action.WARN
    else:
        action = Action.ALLOW

    return DecisionResult(
        action=action,
        risk_score=risk,
        toxicity_score=toxicity_score,
        session_risk=session_risk,
        intent_label=intent_label,
        confidence=confidence,
        trie_matched=False,
        api_escalated=False,
    )


def needs_escalation(confidence: float, trie_matched: bool) -> bool:
    """True if message should be routed to LLM API escalation path."""
    return not trie_matched and confidence < CONFIDENCE_THRESHOLD
