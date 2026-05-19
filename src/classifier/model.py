"""
ModernBERT multi-task model.
Shared encoder → toxicity head (sigmoid) + intent head (softmax).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelOutput:
    toxicity_score: torch.Tensor        # (batch,) float in [0, 1]
    intent_logits: torch.Tensor         # (batch, n_intents)
    toxicity_confidence: torch.Tensor   # (batch,) how far from 0.5
    intent_confidence: torch.Tensor     # (batch,) max softmax probability
    confidence: torch.Tensor            # (batch,) min(tox_conf, intent_conf)
    embedding: torch.Tensor             # (batch, 768) CLS embedding
    loss: Optional[torch.Tensor] = None


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class ContentModerationModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "answerdotai/ModernBERT-large",
        n_intents: int = 21,
        dropout: float = 0.1,
        alpha: float = 0.5,     # toxicity loss weight
        beta: float = 0.5,      # intent loss weight
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(
            encoder_name, 
            config=config, 
            attn_implementation=attn_implementation
        )
        hidden = config.hidden_size  # 1024 for ModernBERT-large

        self.dropout = nn.Dropout(dropout)

        # Toxicity head: binary classification
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Intent head: multi-class classification
        self.intent_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_intents),
        )

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        toxicity_labels: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token

        toxicity_score = self.toxicity_head(cls).squeeze(-1)    # (batch,)
        intent_logits = self.intent_head(cls)                    # (batch, n_intents)

        # Confidence computation (see FYP.md §3 — Confidence Threshold Routing)
        tox_conf = (toxicity_score - 0.5).abs() * 2             # 0=uncertain, 1=certain
        intent_conf = intent_logits.softmax(-1).max(-1).values  # max softmax prob
        confidence = torch.minimum(tox_conf, intent_conf)       # most conservative

        loss = None
        if toxicity_labels is not None and intent_labels is not None:
            tox_loss = self.focal_loss(toxicity_score, toxicity_labels.float())
            # If entire batch has no real intent labels (-100), skip intent loss
            # to avoid NaN from CrossEntropyLoss dividing over zero valid samples
            valid_intent = (intent_labels != -100)
            if valid_intent.any():
                int_loss = self.ce_loss(intent_logits, intent_labels)
            else:
                # Multiply by 0 instead of creating a detached zero tensor —
                # keeps intent_logits in the computation graph so DDP doesn't
                # complain about parameters with no gradient this step
                int_loss = (intent_logits * 0.0).sum()
            loss = self.alpha * tox_loss + self.beta * int_loss

        return ModelOutput(
            toxicity_score=toxicity_score,
            intent_logits=intent_logits,
            toxicity_confidence=tox_conf,
            intent_confidence=intent_conf,
            confidence=confidence,
            embedding=cls,
            loss=loss,
        )

    @torch.no_grad()
    def predict(self, input_ids, attention_mask) -> dict:
        self.eval()
        out = self(input_ids, attention_mask)
        intent_probs = out.intent_logits.softmax(-1)
        return {
            "toxicity_score": out.toxicity_score.cpu().numpy(),
            "intent_probs": intent_probs.cpu().numpy(),
            "intent_label_idx": intent_probs.argmax(-1).cpu().numpy(),
            "confidence": out.confidence.cpu().numpy(),
            "embedding": out.embedding.cpu().numpy(),
        }
