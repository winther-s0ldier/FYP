"""
HMM session model — Baum-Welch training + Viterbi inference.
Computes session risk: P(sequence is abnormal | learned normal model).
Includes noise-aware input handling (confidence-weighted, 'uncertain' state).
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from hmmlearn import hmm
from src.classifier.dataset import INTENT_LABELS


# Intent states: all intent labels + "uncertain" (for low-confidence BERT predictions)
ALL_STATES = INTENT_LABELS + ["uncertain"]
STATE2IDX = {s: i for i, s in enumerate(ALL_STATES)}
IDX2STATE = {i: s for s, i in STATE2IDX.items()}
N_STATES = len(ALL_STATES)

# Session-level labels
SESSION_LABELS = ["benign_session", "suspicious_session", "malicious_session"]


class SessionHMM:
    """
    Two HMMs: one trained on benign sessions, one on malicious.
    Risk = 1 - P(sequence | benign_model) / (P(seq|benign) + P(seq|malicious))
    """

    def __init__(self, n_iter: int = 100, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol = tol
        self.benign_model: Optional[hmm.CategoricalHMM] = None
        self.malicious_model: Optional[hmm.CategoricalHMM] = None

    # --- Training ---

    def fit(self, sessions: list[dict]):
        """
        sessions: list of {"intents": [label, ...], "label": "benign_session"|...}
        Trains benign and malicious HMMs separately.
        """
        benign_seqs = [s["intents"] for s in sessions if s["label"] == "benign_session"]
        malicious_seqs = [s["intents"] for s in sessions
                          if s["label"] in ("suspicious_session", "malicious_session")]

        if not benign_seqs:
            raise ValueError("No benign sessions in training data.")
        if not malicious_seqs:
            raise ValueError("No malicious sessions — add more labeled data.")

        print(f"Training benign HMM on {len(benign_seqs)} sessions...")
        self.benign_model = self._train_hmm(benign_seqs)

        print(f"Training malicious HMM on {len(malicious_seqs)} sessions...")
        self.malicious_model = self._train_hmm(malicious_seqs)

        print("HMM training complete.")

    def _train_hmm(self, sessions: list[list[str]]) -> hmm.CategoricalHMM:
        # Encode sequences
        encoded = [np.array([[STATE2IDX[i]] for i in seq]) for seq in sessions]
        lengths = [len(seq) for seq in encoded]
        X = np.concatenate(encoded)

        model = hmm.CategoricalHMM(
            n_components=N_STATES,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=42,
            verbose=False,
        )
        model.fit(X, lengths)
        return model

    # --- Inference ---

    def compute_risk(self, intent_sequence: list[str]) -> float:
        """
        Return session risk in [0, 1].
        High = sequence unlikely under benign model.
        """
        if not intent_sequence:
            return 0.0
        if self.benign_model is None:
            raise RuntimeError("HMM not trained. Run .fit() first.")

        X = np.array([[STATE2IDX.get(i, STATE2IDX["uncertain"])]
                      for i in intent_sequence])

        try:
            log_prob_benign = self.benign_model.score(X)
            log_prob_malicious = self.malicious_model.score(X)
        except Exception:
            return 0.5  # fallback on degenerate sequence

        # Normalised risk: how much more likely is the malicious model?
        # Clip to [0, 1]
        prob_benign = np.exp(log_prob_benign)
        prob_malicious = np.exp(log_prob_malicious)
        total = prob_benign + prob_malicious + 1e-12
        risk = prob_malicious / total
        return float(np.clip(risk, 0.0, 1.0))

    def viterbi_sequence(self, intent_sequence: list[str]) -> list[str]:
        """Return most likely hidden state sequence via Viterbi (benign model)."""
        if not self.benign_model:
            return intent_sequence
        X = np.array([[STATE2IDX.get(i, STATE2IDX["uncertain"])]
                      for i in intent_sequence])
        _, state_seq = self.benign_model.decode(X, algorithm="viterbi")
        return [IDX2STATE[s] for s in state_seq]

    # --- Persistence ---

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"benign": self.benign_model, "malicious": self.malicious_model}, f)
        print(f"HMM saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            models = pickle.load(f)
        self.benign_model = models["benign"]
        self.malicious_model = models["malicious"]
        return self

    @classmethod
    def from_file(cls, path: Path) -> "SessionHMM":
        obj = cls()
        return obj.load(path)


def build_session_sequences(annotated_data: list[dict]) -> list[dict]:
    """
    Convert annotated sessions to HMM training format.
    Input: [{"messages": [{"text": ..., "intent": ..., "intent_confidence": ...}],
              "session_label": "benign_session"|...}]
    Output: [{"intents": [...], "label": ...}]

    Applies confidence-weighted filtering (uncertain state for low-confidence preds).
    """
    from src.session.store import SESSION_WINDOW
    threshold = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", 0.70))

    sequences = []
    for session in annotated_data:
        intents = []
        for msg in session["messages"][-SESSION_WINDOW:]:  # cap at window size
            conf = msg.get("intent_confidence", 1.0)
            label = msg["intent"] if conf >= threshold else "uncertain"
            intents.append(label)
        sequences.append({"intents": intents, "label": session["session_label"]})
    return sequences


import os
