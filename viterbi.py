"""Token-level LID via Viterbi over (language, region) states.

Transition matrix is region-conditioned: switches between geographically-coherent
pairs (e.g. es-MX <-> en-US) have lower penalty than incoherent ones (es-AR <-> en-IN).
This is where idea #2 (geographic conditioning improves token-level LID) lives.

A "token" here is whatever you pass in — typically whitespace-split words. The
emission score is the per-cell LM log-prob of the token; the transition score
penalizes switching states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from hierarchy import Cell, Hierarchy


@dataclass
class TransitionModel:
    """log P(state_t | state_{t-1}). Sparse, region-aware."""

    stay: float = math.log(0.92)
    same_language_switch: float = math.log(0.05)
    coherent_pair_switch: float = math.log(0.025)
    incoherent_switch: float = math.log(0.005)

    coherent_pairs: set[frozenset[tuple[str, str | None]]] = None

    def __post_init__(self) -> None:
        if self.coherent_pairs is None:
            self.coherent_pairs = set()

    def add_coherent_pair(self, a: tuple[str, str | None], b: tuple[str, str | None]) -> None:
        self.coherent_pairs.add(frozenset({a, b}))

    def score(self, prev: Cell, curr: Cell) -> float:
        if prev.key == curr.key:
            return self.stay
        if prev.language == curr.language:
            return self.same_language_switch
        if frozenset({prev.key, curr.key}) in self.coherent_pairs:
            return self.coherent_pair_switch
        return self.incoherent_switch


def viterbi_decode(
    tokens: list[str],
    hierarchy: Hierarchy,
    transitions: TransitionModel,
) -> list[Cell]:
    if not tokens:
        return []

    cells = hierarchy.cells
    n_states = len(cells)
    n_steps = len(tokens)

    # emission[t][s] = log P(token_t | cell_s) + log_prior(cell_s)
    emission = [
        [
            cells[s].lm.logprob(tokens[t])
            + cells[s].log_prior
            + hierarchy.log_language_prior.get(cells[s].language, 0.0)
            for s in range(n_states)
        ]
        for t in range(n_steps)
    ]

    dp = [[float("-inf")] * n_states for _ in range(n_steps)]
    bp = [[0] * n_states for _ in range(n_steps)]

    for s in range(n_states):
        dp[0][s] = emission[0][s]

    for t in range(1, n_steps):
        for s in range(n_states):
            best_prev, best_score = 0, float("-inf")
            for sp in range(n_states):
                score = dp[t - 1][sp] + transitions.score(cells[sp], cells[s])
                if score > best_score:
                    best_score, best_prev = score, sp
            dp[t][s] = best_score + emission[t][s]
            bp[t][s] = best_prev

    last = max(range(n_states), key=lambda s: dp[-1][s])
    path = [last]
    for t in range(n_steps - 1, 0, -1):
        path.append(bp[t][path[-1]])
    path.reverse()
    return [cells[s] for s in path]
