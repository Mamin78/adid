"""Self-training bootstrap for low-resource (language, region) cells.

Reframed idea #3: instead of unsupervised topic modeling, take a tiny labeled
seed for the target cell and expand it from a larger unlabeled pool via
confidence-thresholded self-training. The char n-gram backbone is well-suited
because it stabilizes faster than parametric models on small data.

This is intentionally simple — one round of pseudo-labeling, plus a held-out
seed check to abort if the expanded model degrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM


@dataclass
class BootstrapConfig:
    order: int = 4
    confidence_threshold: float = 0.85  # posterior prob for the target cell
    max_pseudo_examples: int = 1000
    held_out_frac: float = 0.2


def bootstrap_cell(
    language: str,
    region: str | None,
    seed_texts: list[str],
    unlabeled_pool: Iterable[str],
    hierarchy: Hierarchy,
    config: BootstrapConfig = BootstrapConfig(),
) -> tuple[Cell, float]:
    """Train a leaf cell from a small seed, expanding via self-training.

    Returns (cell, held_out_logprob_delta) — positive delta means expansion helped.
    """
    cutoff = max(1, int(len(seed_texts) * (1 - config.held_out_frac)))
    train_seed, held_out = seed_texts[:cutoff], seed_texts[cutoff:]

    seed_lm = NGramLM(order=config.order).fit(train_seed)
    seed_cell = Cell(language=language, region=region, lm=seed_lm)

    # Score the unlabeled pool using the current hierarchy *plus* this seed cell.
    probe_hier = Hierarchy(cells=hierarchy.cells + [seed_cell])
    probe_hier.log_language_prior = dict(hierarchy.log_language_prior)
    if language not in probe_hier.log_language_prior:
        probe_hier.log_language_prior[language] = 0.0

    pseudo: list[str] = []
    for text in unlabeled_pool:
        if len(pseudo) >= config.max_pseudo_examples:
            break
        _, post = probe_hier.predict_dialect(text)
        if post.get((language, region), 0.0) >= config.confidence_threshold:
            pseudo.append(text)

    expanded_lm = NGramLM(order=config.order).fit(train_seed + pseudo)
    expanded_cell = Cell(language=language, region=region, lm=expanded_lm)

    if held_out:
        seed_score = sum(seed_lm.logprob(t) for t in held_out) / len(held_out)
        exp_score = sum(expanded_lm.logprob(t) for t in held_out) / len(held_out)
        delta = exp_score - seed_score
    else:
        delta = 0.0

    chosen = expanded_cell if delta >= 0 else seed_cell
    return chosen, delta
