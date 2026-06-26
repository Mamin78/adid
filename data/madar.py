"""MADAR Arabic dialect corpus loader.

MADAR-26 Travel: 12,000 sentences × 26 varieties (25 cities + MSA), from the
Basic Traveling Expression Corpus (BTEC). Narrow domain (hotel, transport, food)
but the cleanest city-level Arabic benchmark available.

--- Expected file layout ---

After requesting from https://camel.abudhabi.nyu.edu/madar-corpus/ you get:

    MADAR-Corpus-26/
        MADAR-Corpus-26.train.tsv
        MADAR-Corpus-26.dev.tsv
        MADAR-Corpus-26.test.tsv

Each file is tab-separated, UTF-8:
    sentence_text <TAB> dialect_code

dialect_code is one of (3-letter city abbreviation or MSA):
    ALE  Aleppo      → SY      ALG  Algiers    → DZ
    ALX  Alexandria  → EG      AMM  Amman       → JO
    ASW  Aswan       → EG      BAG  Baghdad     → IQ
    BAS  Basra       → IQ      BEI  Beirut      → LB
    BEN  Benghazi    → LY      CAI  Cairo       → EG
    DAM  Damascus    → SY      DOH  Doha        → QA
    FES  Fez         → MA      JED  Jeddah      → SA
    JER  Jerusalem   → PS      KHA  Khartoum    → SD
    MOS  Mosul       → IQ      MSA  Std Arabic  → (none)
    MUS  Muscat      → OM      RAB  Rabat       → MA
    RIY  Riyadh      → SA      SAL  Salt        → JO
    SAN  Sanaa       → YE      SFX  Sfax        → TN
    TRI  Tripoli     → LY      TUN  Tunis       → TN

--- How to get the data ---

1. Go to https://camel.abudhabi.nyu.edu/madar-corpus/
2. Fill the short request form.
3. You receive a download link within 1-2 days.
4. Unzip and point load_split() at the directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CITY_TO_ISO: dict[str, str] = {
    "ALE": "SY", "ALG": "DZ", "ALX": "EG", "AMM": "JO",
    "ASW": "EG", "BAG": "IQ", "BAS": "IQ", "BEI": "LB",
    "BEN": "LY", "CAI": "EG", "DAM": "SY", "DOH": "QA",
    "FES": "MA", "JED": "SA", "JER": "PS", "KHA": "SD",
    "MOS": "IQ", "MUS": "OM", "RAB": "MA", "RIY": "SA",
    "SAL": "JO", "SAN": "YE", "SFX": "TN", "TRI": "LY",
    "TUN": "TN", "MSA": "MSA",
}


@dataclass
class MADARExample:
    text: str
    city: str        # 3-letter code, e.g. "CAI"
    country: str     # ISO-2 code, e.g. "EG", or "MSA"


def _normalize_city(raw: str) -> str:
    raw = raw.strip().upper()
    return raw if raw in CITY_TO_ISO else raw


def _iter_tsv(path: Path) -> Iterator[MADARExample]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            city = _normalize_city(parts[1])
            country = CITY_TO_ISO.get(city, city)
            yield MADARExample(text=text, city=city, country=country)


def _candidate_paths(root: Path, split: str) -> list[Path]:
    patterns = [
        f"MADAR*{split}.tsv", f"MADAR*{split}.txt",
        f"*{split}.tsv", f"*{split}.txt",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(root.glob(pat)))
    for p in sorted(root.glob("*.tsv")):
        if split in p.name.lower() and p not in candidates:
            candidates.append(p)
    return candidates


def load_split(root: str | Path, split: str = "train") -> list[MADARExample]:
    """Load a MADAR-26 split.

    Args:
        root: Directory containing MADAR-Corpus-26.{split}.tsv files.
        split: 'train', 'dev', or 'test'.

    Returns:
        List of MADARExample (text, city, country).
    """
    root = Path(root)
    candidates = _candidate_paths(root, split)
    if not candidates:
        raise FileNotFoundError(
            f"No MADAR {split} file found under {root}. "
            "Download from https://camel.abudhabi.nyu.edu/madar-corpus/"
        )
    return list(_iter_tsv(candidates[0]))


def documents_by_city(examples: list[MADARExample]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in examples:
        out.setdefault(ex.city, []).append(ex.text)
    return out


def documents_by_country(examples: list[MADARExample]) -> dict[str, list[str]]:
    """Merge city-level examples into country-level buckets (drops MSA)."""
    out: dict[str, list[str]] = {}
    for ex in examples:
        if ex.country != "MSA":
            out.setdefault(ex.country, []).append(ex.text)
    return out
