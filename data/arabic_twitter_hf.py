"""Arabic geo-labeled Twitter data from HuggingFace — weak supervision source.

Dataset: Abdelrahman-Rezk/Arabic_Dialect_Identification
  - 440K tweets collected via Twitter user profile geo-declarations
  - 18 Arab countries (no manual dialect annotation — pure geo metadata)
  - Three splits: train (440,052), validation (9,164), test (8,981)
  - Labels: EG, SA, MA, DZ, TN, LY, SD, IQ, SY, LB, JO, PL, QA, KW, AE, YE, OM, BH

Country note:
  "PL" in this dataset → "PS" (Palestine) everywhere else in this project.
  BH (Bahrain) and SA (Saudi Arabia) are present here but absent from Arap-Tweet 2.0.

Coverage vs NADI 2020 (21 countries):
  Covered (18): DZ, BH, EG, IQ, JO, KW, LB, LY, MA, OM, QA, SA, SD, SY, TN, AE, YE, PS
  Missing  (4): KM, DJ, MR, SO  (Comoros, Djibouti, Mauritania, Somalia — all low-resource)

Why this is the right weak-supervision source for the paper:
  Geo-labels come from Twitter user self-declaration (profile descriptions), NOT from
  human dialect annotation. This is the "geo_twitter" distant-supervision condition,
  analogous to CC-TLD for web text.

--- Usage ---

  # Requires: pip install datasets>=2.0
  from data.arabic_twitter_hf import load_hf, documents_by_country

  docs = documents_by_country(load_hf())          # train split
  docs = documents_by_country(load_hf("test"))    # test split

  # Or cache to disk first to avoid re-downloading:
  from data.arabic_twitter_hf import cache_to_dir, load_from_cache
  cache_to_dir("datasets/arabic_twitter_hf")
  docs = load_from_cache("datasets/arabic_twitter_hf")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_HF_DATASET = "Abdelrahman-Rezk/Arabic_Dialect_Identification"

# Integer label → ISO-2 country code (from dataset ClassLabel definition)
# Label 11 is "PL" in the dataset; we remap to "PS" (ISO 3166-1 for Palestine)
_INT_TO_COUNTRY: dict[int, str] = {
    0: "OM", 1: "SD", 2: "SA",  3: "KW",  4: "QA",  5: "LB",
    6: "JO", 7: "SY", 8: "IQ",  9: "MA", 10: "EG", 11: "PS",
   12: "YE", 13: "BH", 14: "DZ", 15: "AE", 16: "TN", 17: "LY",
}

# "PL" is Palestine in this dataset; the rest of the project uses "PS" (ISO 3166-1)
_LABEL_REMAP: dict[str, str] = {"PL": "PS"}

# All 18 country codes this dataset provides (after remap)
DATASET_COUNTRIES: frozenset[str] = frozenset(_INT_TO_COUNTRY.values())


@dataclass
class ArabicTweetExample:
    text: str
    country: str  # ISO-2 code


def load_hf(
    split: str = "train",
    max_per_country: int | None = None,
) -> list[ArabicTweetExample]:
    """Download and return examples from HuggingFace.

    Args:
        split:           "train", "validation", or "test".
        max_per_country: Cap per country (None = no cap).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pip install datasets>=2.0  (or conda install -c huggingface datasets)"
        ) from e

    ds = load_dataset(_HF_DATASET, split=split, trust_remote_code=True)
    examples: list[ArabicTweetExample] = []
    counts: dict[str, int] = {}

    for row in ds:
        raw = row["label"]
        # Labels are integers in this dataset
        if isinstance(raw, int):
            country = _INT_TO_COUNTRY.get(raw)
        else:
            raw_str = str(raw).upper().strip()
            country = _LABEL_REMAP.get(raw_str, raw_str)
        if not country or country not in DATASET_COUNTRIES:
            continue
        if max_per_country is not None and counts.get(country, 0) >= max_per_country:
            continue
        text: str = str(row["text"]).strip()
        if not text:
            continue
        examples.append(ArabicTweetExample(text=text, country=country))
        counts[country] = counts.get(country, 0) + 1

    return examples


def cache_to_dir(
    out_dir: str | Path,
    split: str = "train",
    max_per_country: int | None = None,
) -> dict[str, int]:
    """Download once and write per-country .txt files for fast reload.

    One tweet per line. Returns {country: n_tweets}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_hf(split=split, max_per_country=max_per_country)

    handles: dict[str, object] = {}
    counts: dict[str, int] = {}
    try:
        for ex in examples:
            if ex.country not in handles:
                handles[ex.country] = (out_dir / f"{ex.country}.txt").open(
                    "w", encoding="utf-8"
                )
                counts[ex.country] = 0
            handles[ex.country].write(ex.text.replace("\n", " ") + "\n")  # type: ignore[attr-defined]
            counts[ex.country] += 1
    finally:
        for fh in handles.values():
            fh.close()  # type: ignore[attr-defined]

    return counts


def load_from_cache(root: str | Path) -> list[ArabicTweetExample]:
    """Load pre-cached per-country .txt files written by cache_to_dir()."""
    root = Path(root)
    examples: list[ArabicTweetExample] = []
    for p in sorted(root.glob("*.txt")):
        country = p.stem.upper()
        if country not in DATASET_COUNTRIES:
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n").strip()
                if text:
                    examples.append(ArabicTweetExample(text=text, country=country))
    return examples


def documents_by_country(
    examples: list[ArabicTweetExample],
) -> dict[str, list[str]]:
    """Group tweet texts by country code."""
    out: dict[str, list[str]] = {}
    for ex in examples:
        if ex.country:
            out.setdefault(ex.country, []).append(ex.text)
    return out


def load_from_dir(root: str | Path) -> list[ArabicTweetExample]:
    """Alias for load_from_cache — load per-country files from a directory."""
    return load_from_cache(root)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cache HF Arabic Twitter dataset to disk.")
    ap.add_argument("--out-dir", required=True, help="Output directory for .txt files.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-per-country", type=int, default=None)
    args = ap.parse_args()

    print(f"Downloading {_HF_DATASET} ({args.split}) → {args.out_dir} ...")
    counts = cache_to_dir(args.out_dir, args.split, args.max_per_country)
    print("\nTweets per country:")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {n:,}")
    print(f"\nTotal: {sum(counts.values()):,} across {len(counts)} countries.")
    print(f"Countries present: {sorted(counts.keys())}")
    print(f"NADI countries missing: {sorted({'KM','DJ','MR','SO'} - set(counts.keys()))}")
