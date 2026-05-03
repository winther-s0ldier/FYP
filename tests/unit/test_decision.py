"""Unit tests for the decision engine."""
import pytest
from src.pipeline.decision import (
    decide_fast_path, needs_escalation, fuse_risk,
    Action, BLOCK_THRESHOLD, FLAG_THRESHOLD, WARN_THRESHOLD
)


def test_trie_match_always_blocks():
    result = decide_fast_path(0.1, 0.0, "greeting", 0.9, 0, trie_matched=True)
    assert result.action == Action.BLOCK
    assert result.risk_score == 1.0


def test_high_toxicity_blocks():
    result = decide_fast_path(0.95, 0.9, "threat", 0.9, 1, trie_matched=False)
    assert result.action == Action.BLOCK


def test_medium_toxicity_flags():
    result = decide_fast_path(0.70, 0.50, "harassment", 0.85, 0, trie_matched=False)
    assert result.action in (Action.FLAG, Action.BLOCK)


def test_low_toxicity_allows():
    result = decide_fast_path(0.05, 0.0, "greeting", 0.95, 0, trie_matched=False)
    assert result.action == Action.ALLOW


def test_escalation_triggered_low_confidence():
    assert needs_escalation(confidence=0.40, trie_matched=False) is True


def test_no_escalation_high_confidence():
    assert needs_escalation(confidence=0.85, trie_matched=False) is False


def test_no_escalation_trie_match():
    # Trie match bypasses escalation
    assert needs_escalation(confidence=0.30, trie_matched=True) is False


def test_fuse_risk_weights():
    risk = fuse_risk(toxicity_score=0.8, session_risk=0.6, vector_hits=1)
    assert 0 < risk <= 1.0
    # Vector hit adds W_VECTOR=0.15 on top
    risk_no_vec = fuse_risk(toxicity_score=0.8, session_risk=0.6, vector_hits=0)
    assert risk > risk_no_vec
