"""Character n-gram language model with modified Kneser-Ney smoothing.

This is the leaf scorer in the hierarchy: one of these per (language, region) cell.
Kept dependency-free so the whole stack stays tiny.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


BOS = "\x02"
EOS = "\x03"


def _pad(text: str, order: int) -> str:
    return BOS * (order - 1) + text + EOS


def _ngrams(text: str, order: int) -> Iterable[str]:
    padded = _pad(text, order)
    for i in range(len(padded) - order + 1):
        yield padded[i : i + order]


@dataclass
class NGramLM:
    order: int = 4
    discount: float = 0.75
    ngram_counts: list[Counter] = field(default_factory=list)
    context_totals: list[dict] = field(default_factory=list)
    continuation_counts: list[dict] = field(default_factory=list)
    vocab: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.ngram_counts:
            self.ngram_counts = [Counter() for _ in range(self.order)]
            self.context_totals = [defaultdict(int) for _ in range(self.order)]
            self.continuation_counts = [defaultdict(set) for _ in range(self.order)]

    def fit(self, texts: Iterable[str]) -> "NGramLM":
        for text in texts:
            self._observe(text)
        return self

    def _observe(self, text: str) -> None:
        for ch in text:
            self.vocab.add(ch)
        for n in range(1, self.order + 1):
            for gram in _ngrams(text, n):
                self.ngram_counts[n - 1][gram] += 1
                if n > 1:
                    ctx, w = gram[:-1], gram[-1]
                    self.context_totals[n - 1][ctx] += 1
                    self.continuation_counts[n - 1][ctx].add(w)

    def logprob(self, text: str) -> float:
        """Sum of log P(char_i | context) over the text."""
        total = 0.0
        for gram in _ngrams(text, self.order):
            total += math.log(self._prob_with_backoff(gram, self.order))
        return total

    def _prob_with_backoff(self, gram: str, n: int) -> float:
        if n == 1:
            # unigram: add-one over observed vocab
            v = max(len(self.vocab), 1)
            c = self.ngram_counts[0][gram[-1]]
            total = sum(self.ngram_counts[0].values())
            return (c + 1.0) / (total + v)

        ctx, w = gram[:-1], gram[-1]
        ctx_total = self.context_totals[n - 1].get(ctx, 0)
        if ctx_total == 0:
            return self._prob_with_backoff(gram[1:], n - 1)

        c = self.ngram_counts[n - 1][gram]
        discounted = max(c - self.discount, 0.0) / ctx_total
        unique_follows = len(self.continuation_counts[n - 1].get(ctx, ()))
        lam = (self.discount * unique_follows) / ctx_total
        return discounted + lam * self._prob_with_backoff(gram[1:], n - 1)

    def perplexity(self, text: str) -> float:
        if not text:
            return float("inf")
        return math.exp(-self.logprob(text) / max(len(text), 1))
