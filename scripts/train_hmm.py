"""
Train HMM session model on labeled conversation sequences.
Run after: scripts/annotate_sessions.py (or place custom data in data/custom/)

Usage: python scripts/train_hmm.py
"""
import json
from pathlib import Path
from src.session.hmm import SessionHMM, build_session_sequences

DATA_PATH = Path("data/custom/session_sequences.jsonl")
SAVE_PATH = Path("models/checkpoints/hmm.pkl")


def main():
    if not DATA_PATH.exists():
        print(f"Session data not found at {DATA_PATH}")
        print("Creating a small synthetic dataset for testing...")
        _create_synthetic_data()

    print(f"Loading session data from {DATA_PATH}...")
    sessions_raw = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions_raw.append(json.loads(line))

    print(f"Loaded {len(sessions_raw)} sessions.")

    # Convert to HMM training format
    sessions = build_session_sequences(sessions_raw)

    label_counts = {}
    for s in sessions:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    print("Session label distribution:", label_counts)

    # Train
    hmm = SessionHMM(n_iter=100, tol=1e-4)
    hmm.fit(sessions)

    # Save
    hmm.save(SAVE_PATH)
    print(f"\nHMM saved to {SAVE_PATH}")

    # Quick sanity check
    test_cases = [
        (["greeting", "question", "farewell"], "benign"),
        (["greeting", "personal_probe", "threat"], "malicious"),
        (["greeting", "question"] * 5, "benign"),
    ]
    print("\nSanity checks:")
    for seq, expected in test_cases:
        risk = hmm.compute_risk(seq)
        print(f"  {seq[:3]}... → risk={risk:.3f} (expected: {'high' if expected=='malicious' else 'low'})")


def _create_synthetic_data():
    """Create minimal synthetic sessions for initial testing."""
    import random
    random.seed(42)

    benign_patterns = [
        ["greeting", "question", "information_request", "farewell"],
        ["greeting", "small_talk", "help_request", "question", "farewell"],
        ["question", "feedback", "farewell"],
    ]
    malicious_patterns = [
        ["greeting", "question", "personal_probe", "threat"],
        ["small_talk", "personal_probe", "unusual_urgency", "threat"],
        ["greeting", "question", "grooming_signal", "doxxing_attempt"],
        ["greeting"] * 5 + ["threat"],  # warmup-then-attack
    ]

    sessions = []
    for _ in range(100):
        pattern = random.choice(benign_patterns)
        sessions.append({
            "messages": [{"intent": i, "intent_confidence": 0.85, "text": ""} for i in pattern],
            "session_label": "benign_session"
        })
    for _ in range(60):
        pattern = random.choice(malicious_patterns)
        sessions.append({
            "messages": [{"intent": i, "intent_confidence": 0.85, "text": ""} for i in pattern],
            "session_label": "malicious_session"
        })

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    print(f"Created synthetic data: {len(sessions)} sessions → {DATA_PATH}")


if __name__ == "__main__":
    main()
