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

        # hmmlearn sizes emissionprob_ to (n_components, max_obs_seen+1).
        # If an intent label like "grooming" (idx=16) never appears in the
        # training split, the matrix has only e.g. 8 columns → IndexError when
        # scoring sequences that contain those unseen intents.
        # Fix: expand to the full vocabulary width first.
        n_trained = model.emissionprob_.shape[1]
        if n_trained < N_STATES:
            pad = np.zeros((model.n_components, N_STATES - n_trained))
            model.emissionprob_ = np.hstack([model.emissionprob_, pad])
            model.n_features = N_STATES
            
        # Laplace smoothing: fills the padded zeros and prevents log(0)=-inf
        # for intents absent from one class's training data.
        eps = 1e-6
        model.emissionprob_ = model.emissionprob_ + eps
        model.emissionprob_ /= model.emissionprob_.sum(axis=1, keepdims=True)

        return model

    # --- Inference ---

    def compute_risk(self, intent_sequence: list[str], window: int = 8) -> float:
        """
        Return session risk in [0, 1].
        High = sequence unlikely under benign model.

        window: only the last `window` intents are scored.
        Prevents benign history from diluting a sudden late-session threat.
        Default 8 — long enough to capture escalation patterns, short enough
        that a single threatening turn in a previously friendly session still
        pushes risk above 0.5.
        """
        if not intent_sequence:
            return 0.0
        if self.benign_model is None:
            raise RuntimeError("HMM not trained. Run .fit() first.")

        # Sliding window: cap at last `window` turns
        intent_sequence = intent_sequence[-window:]

        X = np.array([[STATE2IDX.get(i, STATE2IDX["uncertain"])]
                      for i in intent_sequence])

        try:
            log_prob_benign = self.benign_model.score(X)
            log_prob_malicious = self.malicious_model.score(X)
        except Exception as e:
            # score() can raise if hmmlearn's forward pass hits a degenerate state.
            # Laplace smoothing in _train_hmm should prevent this; if it still
            # fires something is deeply wrong — treat as ambiguous, not benign.
            print(f"[HMM] score() failed: {e}. Returning 0.5 (ambiguous).")
            return 0.5

        # Handle -inf: if one model can't score the sequence at all, assign
        # full credit to the other.
        if np.isneginf(log_prob_malicious) and np.isneginf(log_prob_benign):
            return 0.5   # both confused — treat as ambiguous
        if np.isneginf(log_prob_malicious):
            return 0.0   # malicious model has no explanation → benign
        if np.isneginf(log_prob_benign):
            return 1.0   # benign model has no explanation → very suspicious

        # Log-space computation using log-sum-exp trick — avoids float underflow.
        # np.exp(-100) = 0.0, so naive division fails for long/rare sequences.
        # logaddexp(a, b) = log(exp(a) + exp(b)) computed stably.
        # risk = exp(log_mal - log(exp(log_ben) + exp(log_mal)))
        log_total = np.logaddexp(log_prob_benign, log_prob_malicious)
        risk = float(np.exp(log_prob_malicious - log_total))
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
