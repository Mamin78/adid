"""NADI (Nuanced Arabic Dialect Identification) shared-task data loader.

Covers NADI 2020 and 2021 subtask 1: country-level tweet classification.

--- Expected file layout ---

After downloading from the shared-task site, each release contains:
    NADI2020_train.tsv   (or train.tsv / NADI_train.tsv — all accepted)
    NADI2020_dev.tsv
    NADI2020_test.tsv    (gold labels released post-task)

Each file is tab-separated, UTF-8, NO header row:
    tweet_id <TAB> tweet_text <TAB> country_label

country_label is one of:
    Algeria, Bahrain, Egypt, Iraq, Jordan, Kuwait, Lebanon, Libya,
    Mauritania, Morocco, Oman, Palestine, Qatar, Saudi_Arabia,
    Somalia, Sudan, Syria, Tunisia, UAE, Yemen
    (some releases use "Saudi Arabia" with a space, or ISO codes; both handled)

--- How to get the data ---

1. Go to https://sites.google.com/view/nadi-shared-task/
2. Fill the access form (approved immediately).
3. Download NADI 2020 Subtask 1 (country-level).
4. Unzip into a directory, e.g. datasets/nadi2020/.
5. Pass that directory to load_split().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Canonical 2-letter ISO country codes used as internal labels.
# Maps every label variant found across NADI releases → ISO code.
LABEL_TO_ISO: dict[str, str] = {
    # Full names (NADI 2020/2021 style)
    "Algeria": "DZ", "Bahrain": "BH", "Comoros": "KM", "Djibouti": "DJ",
    "Egypt": "EG", "Iraq": "IQ", "Jordan": "JO", "Kuwait": "KW",
    "Lebanon": "LB", "Libya": "LY", "Mauritania": "MR", "Morocco": "MA",
    "Oman": "OM", "Palestine": "PS", "Qatar": "QA",
    "Saudi_Arabia": "SA", "Saudi Arabia": "SA",
    "Somalia": "SO", "Sudan": "SD", "Syria": "SY",
    "Tunisia": "TN", "UAE": "AE",
    "United Arab Emirates": "AE", "United_Arab_Emirates": "AE",
    "Yemen": "YE",
}
# Add ISO-to-ISO pass-through so callers can use either form.
LABEL_TO_ISO.update({v: v for v in set(LABEL_TO_ISO.values())})

ISO_TO_LABEL: dict[str, str] = {
    "DZ": "Algeria", "BH": "Bahrain", "KM": "Comoros", "DJ": "Djibouti",
    "EG": "Egypt", "IQ": "Iraq", "JO": "Jordan", "KW": "Kuwait",
    "LB": "Lebanon", "LY": "Libya", "MR": "Mauritania", "MA": "Morocco",
    "OM": "Oman", "PS": "Palestine", "QA": "Qatar", "SA": "Saudi Arabia",
    "SO": "Somalia", "SD": "Sudan", "SY": "Syria", "TN": "Tunisia",
    "AE": "UAE", "YE": "Yemen",
}


@dataclass
class NADIExample:
    tweet_id: str
    text: str
    country: str  # 2-letter ISO code, or "" if test set (no label)


def _normalize_label(raw: str) -> str:
    raw = raw.strip()
    if raw in LABEL_TO_ISO:
        return LABEL_TO_ISO[raw]
    # Try case-insensitive lookup
    for k, v in LABEL_TO_ISO.items():
        if k.lower() == raw.lower():
            return v
    return raw  # return as-is if unknown; caller can filter


def _candidate_paths(root: Path, split: str) -> list[Path]:
    """Generate likely file paths for a split name."""
    patterns = [
        f"NADI*_{split}.tsv", f"NADI*_{split}.txt",
        f"{split}.tsv", f"{split}.txt",
        f"nadi*_{split}.tsv",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(root.glob(pat)))
    # Fallback: any TSV whose stem contains the split name
    for p in sorted(root.glob("*.tsv")):
        if split in p.stem.lower() and p not in candidates:
            candidates.append(p)
    return candidates


def _iter_tsv(path: Path) -> Iterator[NADIExample]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Skip header row (starts with '#' in NADI 2020 format)
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tweet_id = parts[0].strip()
            text = parts[1].strip()
            country = _normalize_label(parts[2]) if len(parts) >= 3 else ""
            yield NADIExample(tweet_id=tweet_id, text=text, country=country)


def load_split(root: str | Path, split: str = "train") -> list[NADIExample]:
    """Load a NADI split from a directory.

    Args:
        root: Directory containing the TSV files.
        split: One of 'train', 'dev', 'test'.

    Returns:
        List of NADIExample. Examples without a label (test set) have country=''.
    """
    root = Path(root)
    candidates = _candidate_paths(root, split)
    if not candidates:
        raise FileNotFoundError(
            f"No NADI {split} file found under {root}. "
            "Expected a TSV named like NADI2020_train.tsv or train.tsv.\n"
            "Download from https://sites.google.com/view/nadi-shared-task/"
        )
    return list(_iter_tsv(candidates[0]))


def documents_by_country(examples: list[NADIExample]) -> dict[str, list[str]]:
    """Group tweet texts by country ISO code. Unlabeled examples are skipped."""
    out: dict[str, list[str]] = {}
    for ex in examples:
        if ex.country:
            out.setdefault(ex.country, []).append(ex.text)
    return out


def label_set(examples: list[NADIExample]) -> list[str]:
    """Sorted list of country codes present in this split."""
    return sorted({ex.country for ex in examples if ex.country})
