"""Arap-Tweet loader — geo-labeled Arabic tweets for weak supervision.

Arap-Tweet 2.0 (Zaghouani & Charfi, 2018) is a corpus of ~5 million tweets
from 17 Arab countries, balanced by country/gender/age. It is the primary
source of per-country training data for the geo-n-gram experiments.

--- Expected file layouts ---

The dataset has been distributed in two forms; both are supported.

Form A — directory of per-country TSV files (preferred):
    <root>/
        EG.tsv   (tweet_id <TAB> tweet_text)
        SA.tsv
        MA.tsv
        ...
    Country code is inferred from the filename stem (EG, SA, MA, ...).

Form B — single TSV with a country column:
    <root>/aratweet.tsv
    tweet_id <TAB> tweet_text <TAB> country_code

--- How to get the data ---

Option 1 (original corpus):
    Email the authors (Zaghouani et al.) via the contact on the paper page:
    https://arxiv.org/abs/1808.07674
    Mention you're using it for NLP research.

Option 2 (re-shared on HuggingFace / OSIAN):
    Search HuggingFace Datasets for "arap-tweet" or "arabic-dialect-tweets".
    Several community uploads exist with the same structure.

Option 3 (reconstruct from Twitter IDs):
    If you have only tweet IDs, use twarc2 to re-hydrate.
    This may yield a partial corpus due to deleted tweets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from data.nadi import LABEL_TO_ISO

# Arap-Tweet uses 17 of the 21 NADI countries; same ISO codes.
_KNOWN_STEMS = {
    "EG", "SA", "MA", "DZ", "TN", "LY", "SD", "IQ", "SY", "LB",
    "JO", "PS", "QA", "KW", "AE", "YE", "OM",
}


@dataclass
class ArapTweetExample:
    tweet_id: str
    text: str
    country: str  # 2-letter ISO code


def _country_from_stem(stem: str) -> str | None:
    upper = stem.upper()
    if upper in _KNOWN_STEMS:
        return upper
    return LABEL_TO_ISO.get(upper)


def _load_per_country_dir(root: Path) -> list[ArapTweetExample]:
    examples: list[ArapTweetExample] = []
    for p in sorted(root.glob("*.tsv")):
        country = _country_from_stem(p.stem)
        if country is None:
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                tweet_id = parts[0].strip()
                text = parts[1].strip() if len(parts) >= 2 else ""
                if text:
                    examples.append(ArapTweetExample(tweet_id=tweet_id, text=text, country=country))
    return examples


def _load_single_tsv(path: Path) -> list[ArapTweetExample]:
    examples: list[ArapTweetExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            tweet_id = parts[0].strip()
            text = parts[1].strip()
            raw_country = parts[2].strip()
            country = LABEL_TO_ISO.get(raw_country, raw_country)
            if text and country:
                examples.append(ArapTweetExample(tweet_id=tweet_id, text=text, country=country))
    return examples


def load(root: str | Path) -> list[ArapTweetExample]:
    """Load Arap-Tweet from a directory.

    Auto-detects layout:
      - If the directory contains *.tsv files named by country code → Form A.
      - If it contains a single large TSV with 3+ columns → Form B.
    """
    root = Path(root)
    country_files = [p for p in root.glob("*.tsv") if _country_from_stem(p.stem) is not None]
    if country_files:
        return _load_per_country_dir(root)

    single_files = list(root.glob("*.tsv"))
    if single_files:
        return _load_single_tsv(single_files[0])

    raise FileNotFoundError(
        f"No Arap-Tweet data found under {root}. "
        "Expected either per-country TSV files (EG.tsv, SA.tsv, ...) "
        "or a single TSV with columns: tweet_id, text, country_code. "
        "See module docstring for acquisition instructions."
    )


def documents_by_country(examples: list[ArapTweetExample]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in examples:
        if ex.country:
            out.setdefault(ex.country, []).append(ex.text)
    return out
