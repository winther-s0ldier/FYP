"""
Adversarial session manipulation tests (FYP.md §12.3).
Tests HMM robustness to warmup attacks, alternating patterns, noise.
"""
import pytest
from src.session.hmm import SessionHMM, ALL_STATES


@pytest.fixture
def trained_hmm(tmp_path):
    """Build a minimal trained HMM using synthetic sessions."""
    hmm = SessionHMM(n_iter=20)
    benign = [
        {"intents": ["greeting", "question", "question", "farewell"], "label": "benign_session"},
        {"intents": ["greeting", "small_talk", "information_request"], "label": "benign_session"},
        {"intents": ["help_request", "question", "feedback", "farewell"], "label": "benign_session"},
    ] * 10  # repeat for enough data

    malicious = [
        {"intents": ["greeting", "question", "personal_probe", "threat"], "label": "malicious_session"},
        {"intents": ["small_talk", "personal_probe", "unusual_urgency", "threat"], "label": "malicious_session"},
        {"intents": ["greeting", "question", "grooming_signal", "doxxing_attempt"], "label": "malicious_session"},
    ] * 10

    hmm.fit(benign + malicious)
    return hmm


def test_warmup_then_attack(trained_hmm):
    """
    Adversary sends 10 friendly messages to build trust, then attacks.
    HMM must still flag due to the transition to threat.
    """
    warmup = ["greeting", "question", "small_talk", "feedback",
              "question", "greeting", "small_talk", "help_request",
              "question", "farewell"]
    attack = warmup + ["threat"]

    warmup_risk = trained_hmm.compute_risk(warmup)
    attack_risk = trained_hmm.compute_risk(attack)

    # Risk should increase significantly after the threat
    assert attack_risk > warmup_risk, (
        f"HMM failed to catch warmup attack: warmup_risk={warmup_risk:.3f}, "
        f"attack_risk={attack_risk:.3f}"
    )


def test_alternating_risk_accumulates(trained_hmm):
    """
    Alternating safe/toxic messages. Cumulative risk should reflect the pattern.
    """
    alternating = []
    for _ in range(5):
        alternating.extend(["small_talk", "threat"])

    risk = trained_hmm.compute_risk(alternating)
    benign_baseline = trained_hmm.compute_risk(["greeting", "question"] * 5)

    assert risk > benign_baseline, (
        f"Alternating pattern not flagged: risk={risk:.3f}, baseline={benign_baseline:.3f}"
    )


def test_hmm_noise_robustness(trained_hmm):
    """
    BERT misclassifies 2/10 friendly intents as 'uncertain'.
    Threat at the end should still be caught.
    """
    noisy_sequence = [
        "greeting", "question", "uncertain",  # uncertain = BERT low confidence
        "question", "small_talk", "uncertain",
        "question", "question",
        "personal_probe", "threat"
    ]
    benign_clean = ["greeting", "question", "small_talk",
                    "question", "small_talk", "help_request",
                    "question", "question", "farewell", "farewell"]

    noisy_risk = trained_hmm.compute_risk(noisy_sequence)
    benign_risk = trained_hmm.compute_risk(benign_clean)

    assert noisy_risk > benign_risk, (
        f"Noisy session not flagged over benign: noisy={noisy_risk:.3f}, benign={benign_risk:.3f}"
    )


def test_pure_benign_low_risk(trained_hmm):
    """A clearly benign session should have low risk score."""
    benign = ["greeting", "question", "information_request", "feedback", "farewell"]
    risk = trained_hmm.compute_risk(benign)
    assert risk < 0.5, f"Benign session has too-high risk: {risk:.3f}"


def test_empty_session_zero_risk(trained_hmm):
    assert trained_hmm.compute_risk([]) == 0.0
