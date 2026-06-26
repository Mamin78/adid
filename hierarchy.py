"""Hierarchical (language, region) LID over leaf n-gram LMs.

The tree is two-level: language -> region. Scoring uses a Bayesian backoff:

    log P(cell | text) ∝ log P(text | cell) + log P(cell)
    log P(cell) = log P(region | language) + log P(language)

For dialect ID, we marginalize over regions within a language.
For LID, we marginalize over all regions.

This is where idea #1 (country-conditioned n-grams) lives architecturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from ngram_lm import NGramLM


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass
class Cell:
    """A (language, region) leaf. region=None means region-agnostic fallback."""

    language: str
    region: str | None
    lm: NGramLM
    log_prior: float = 0.0  # log P(region | language); 0 for region=None

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.language, self.region)


@dataclass
class Hierarchy:
    cells: list[Cell] = field(default_factory=list)
    log_language_prior: dict[str, float] = field(default_factory=dict)

    def add_cell(self, cell: Cell) -> None:
        self.cells.append(cell)

    def fit_priors_from_counts(self, counts: dict[tuple[str, str | None], int]) -> None:
        """Set priors from a (language, region) -> doc_count dictionary."""
        lang_totals: dict[str, int] = {}
        for (lang, _region), c in counts.items():
            lang_totals[lang] = lang_totals.get(lang, 0) + c
        grand = sum(lang_totals.values())

        for lang, total in lang_totals.items():
            self.log_language_prior[lang] = math.log(total / grand)

        for cell in self.cells:
            n_lr = counts.get(cell.key, 0)
            denom = lang_totals.get(cell.language, 0)
            if denom == 0 or n_lr == 0:
                cell.log_prior = float("-inf") if cell.region is not None else 0.0
            else:
                cell.log_prior = math.log(n_lr / denom)

    def score_cells(self, text: str) -> list[tuple[Cell, float]]:
        """Return (cell, log-joint) pairs for every cell."""
        out = []
        for cell in self.cells:
            lp_text = cell.lm.logprob(text)
            lp_region = cell.log_prior
            lp_lang = self.log_language_prior.get(cell.language, 0.0)
            out.append((cell, lp_text + lp_region + lp_lang))
        return out

    def predict_language(self, text: str) -> tuple[str, dict[str, float]]:
        """Marginalize over regions to get language posterior."""
        scored = self.score_cells(text)
        by_lang: dict[str, list[float]] = {}
        for cell, lp in scored:
            by_lang.setdefault(cell.language, []).append(lp)
        marg = {lang: _logsumexp(lps) for lang, lps in by_lang.items()}
        z = _logsumexp(list(marg.values()))
        posterior = {lang: math.exp(lp - z) for lang, lp in marg.items()}
        best = max(posterior, key=posterior.get)
        return best, posterior

    def predict_dialect(self, text: str) -> tuple[tuple[str, str | None], dict]:
        """Predict full (language, region) cell."""
        scored = self.score_cells(text)
        z = _logsumexp([lp for _, lp in scored])
        posterior = {cell.key: math.exp(lp - z) for cell, lp in scored}
        best = max(posterior, key=posterior.get)
        return best, posterior
