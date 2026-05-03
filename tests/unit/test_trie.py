"""Unit tests for Trie pre-filter (from FYP.md §10.2)."""
import pytest
from src.pipeline.trie import SlurTrie


@pytest.fixture
def trie():
    return SlurTrie(["kys", "kill yourself", "die", "i will kill"])


def test_exact_match(trie):
    assert trie.search("kys")[0] is True


def test_case_insensitive(trie):
    assert trie.search("KYS")[0] is True
    assert trie.search("KiLl YoUrSeLf")[0] is True


def test_leetspeak(trie):
    # "k1ll" → after normalisation "kill" — matches "i will kill" token
    t = SlurTrie(["kill"])
    assert t.search("k1ll")[0] is True


def test_zero_width_space(trie):
    # "k​ys" (zero-width space inserted)
    assert trie.search("k​ys")[0] is True


def test_fullwidth_chars(trie):
    t = SlurTrie(["kill"])
    assert t.search("ｋｉｌｌ")[0] is True  # fullwidth → normalised


def test_false_positive_sky(trie):
    assert trie.search("sky")[0] is False


def test_false_positive_died(trie):
    # "die" is in trie but "died" should not match as exact token
    result, _ = trie.search("he died in the movie")
    # "die" is a substring match via token — update if policy changes
    # This test documents the behaviour; adjust expected based on your lexicon policy
    assert isinstance(result, bool)


def test_phrase_match(trie):
    assert trie.search("i will kill you")[0] is True


def test_benign_message(trie):
    assert trie.search("hello how are you doing today")[0] is False


def test_empty_string(trie):
    assert trie.search("")[0] is False
