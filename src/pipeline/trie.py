"""
Trie pre-filter — O(L) exact match for slurs/keywords before BERT inference.
Handles case normalisation, basic l33tspeak, zero-width spaces.
"""
import re
import unicodedata


LEETSPEAK_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
    "!": "i", "+": "t",
}

# Default seed list — extend with domain-specific lexicon
DEFAULT_SLURS: list[str] = [
    "kys", "kill yourself", "kms",
    "die", "i will kill", "i'll kill",
    "rape", "molest",
    # Add more from a curated lexicon file
]


class TrieNode:
    __slots__ = ("children", "is_end", "word")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.word: str = ""


class SlurTrie:
    def __init__(self, words: list[str] | None = None):
        self.root = TrieNode()
        for w in (words or DEFAULT_SLURS):
            self.insert(w)

    # --- Build ---

    def insert(self, word: str):
        node = self.root
        for ch in self._normalise(word):
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        node.word = word

    # --- Query ---

    def search(self, text: str) -> tuple[bool, str]:
        """Return (matched, matched_word). O(|text| * L)."""
        normalised = self._normalise(text)
        tokens = normalised.split()
        # Check full text (for multi-word phrases)
        if self._match_phrase(normalised):
            return True, normalised
        # Check each token
        for token in tokens:
            if self._match_phrase(token):
                return True, token
        return False, ""

    def _match_phrase(self, text: str) -> bool:
        node = self.root
        for ch in text:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    # --- Normalisation ---

    @staticmethod
    def _normalise(text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove zero-width and invisible characters
        text = re.sub(r"[​-‏‪-‮﻿]", "", text)
        # Normalise unicode (fullwidth → ASCII)
        text = unicodedata.normalize("NFKC", text)
        # L33tspeak substitution
        for leet, normal in LEETSPEAK_MAP.items():
            text = text.replace(leet, normal)
        # Strip punctuation (keep spaces for phrase matching)
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_from_file(self, path: str):
        """Load one word/phrase per line from a lexicon file."""
        with open(path) as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    self.insert(word)
        return self
