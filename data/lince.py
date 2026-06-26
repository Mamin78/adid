"""LinCE Spanish-English LID loader.

LinCE distributes token-level LID data in CoNLL-style format: one token per
line, tab-separated, blank line between sentences. The Spa-Eng LID task uses
these labels (per the LinCE paper, Aguilar et al. 2020):

    lang1   -- Spanish
    lang2   -- English
    other   -- punctuation, numbers, emoji
    ne      -- named entities
    ambiguous
    fw      -- foreign word (rare)
    mixed   -- intra-word mixing (rare)
    unk

For tinylid we collapse to {es, en, other}: lang1->es, lang2->en, everything
else->other. That keeps the eval comparable to a 2-language LID model while
not punishing the model for token types it was never trained on.

Test set labels are not released (it's a leaderboard); use dev for offline eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LABEL_MAP = {
    "lang1": "es",
    "lang2": "en",
    "other": "other",
    "ne": "other",
    "ambiguous": "other",
    "fw": "other",
    "mixed": "other",
    "unk": "other",
}


@dataclass
class TaggedSentence:
    tokens: list[str]
    labels: list[str]  # mapped to {es, en, other}
    raw_labels: list[str]  # original LinCE labels


def _iter_conll(path: Path) -> Iterator[TaggedSentence]:
    tokens: list[str] = []
    raw: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield TaggedSentence(
                        tokens=tokens,
                        labels=[LABEL_MAP.get(r, "other") for r in raw],
                        raw_labels=raw,
                    )
                tokens, raw = [], []
                continue
            if line.startswith("# "):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                # Some LinCE files have token-only lines for test; skip cleanly.
                continue
            tokens.append(parts[0])
            raw.append(parts[1])
    if tokens:
        yield TaggedSentence(
            tokens=tokens,
            labels=[LABEL_MAP.get(r, "other") for r in raw],
            raw_labels=raw,
        )


def load_split(root: str | Path, split: str) -> list[TaggedSentence]:
    """Load a LinCE LID Spa-Eng split.

    Expected layout after unzipping the official archive:
        <root>/lid_spaeng/train.conll
        <root>/lid_spaeng/dev.conll
        <root>/lid_spaeng/test.conll     # tokens only, no labels
    """
    root = Path(root)
    candidates = [
        root / "lid_spaeng" / f"{split}.conll",
        root / f"{split}.conll",
        root / "lid_spaeng" / split,
    ]
    for p in candidates:
        if p.exists():
            return list(_iter_conll(p))
    raise FileNotFoundError(
        f"Could not find LinCE LID Spa-Eng {split} split under {root}. "
        f"Tried: {[str(c) for c in candidates]}. "
        f"Did you run download_lince.py?"
    )


def documents_by_language(sentences: list[TaggedSentence]) -> dict[str, list[str]]:
    """Concatenate runs of same-label tokens into pseudo-documents per language.

    This gives us doc-level training text for the per-language LMs. We only
    keep es/en runs of length >= 3 tokens, to avoid teaching the LM on isolated
    function words mislabeled by the annotators.
    """
    out: dict[str, list[str]] = {"es": [], "en": []}
    for sent in sentences:
        run_lang: str | None = None
        run: list[str] = []
        for tok, lab in zip(sent.tokens, sent.labels):
            if lab == run_lang:
                run.append(tok)
            else:
                if run_lang in out and len(run) >= 3:
                    out[run_lang].append(" ".join(run))
                run_lang, run = lab, [tok]
        if run_lang in out and len(run) >= 3:
            out[run_lang].append(" ".join(run))
    return out
