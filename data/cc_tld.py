"""Common Crawl / OSCAR TLD-filtering pipeline — multi-language.

Extracts per-country text for any supported language from OSCAR 22.01
by filtering documents whose source URL has a country ccTLD for that language.

Supported languages: Arabic (ar), Spanish (es), Portuguese (pt), French (fr).
Add more by editing data/geo_langs.py.

--- Why OSCAR instead of raw Common Crawl ---

Raw CC WARC files are ~50TB per crawl. OSCAR 22.01 is a pre-deduplicated,
language-filtered extraction that is accessible via HuggingFace Datasets and
includes the source URL as metadata.

--- Usage ---

Stream and write to per-country files (one-time, takes hours).
Default corpus is mc4 (public, no login required):

    python -m data.cc_tld --language es --out-dir datasets/cc_es --max-per-country 100000
    python -m data.cc_tld --language pt --out-dir datasets/cc_pt --max-per-country 100000
    python -m data.cc_tld --language ar --out-dir datasets/cc_ar --max-per-country 100000

Use OSCAR (requires HF login: huggingface-cli login):

    python -m data.cc_tld --language ar --corpus oscar --out-dir datasets/cc_ar

Load pre-written files:

    from data.cc_tld import load_from_dir
    docs = load_from_dir("datasets/cc_es")   # → {"MX": [...], "AR": [...], ...}

--- Coverage notes ---

Arabic  (.eg/.sa/.ma/...): Egypt and Saudi Arabia have most content.
Spanish (.mx/.ar/.es/...): Mexico, Argentina, Spain dominate.
Portuguese (.br/.pt/...):  Brazil >> Portugal (~95%/5% split).
French  (.fr/.be/...):     France dominates; .be/.ca/.ch are bilingual/noisy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from data.geo_langs import LanguageConfig, get as get_lang


def _build_tld_regex(lang_cfg: LanguageConfig) -> re.Pattern:
    tlds = sorted(lang_cfg.tld_map.keys(), key=len, reverse=True)
    return re.compile(
        r"\.(" + "|".join(re.escape(t) for t in tlds) + r")(?:/|$)",
        re.IGNORECASE | re.UNICODE,
    )


def tld_to_country(url: str, lang: str | LanguageConfig = "ar") -> str | None:
    """Return an ISO-2 country code if the URL has a TLD for the given language."""
    if not url:
        return None
    cfg = lang if isinstance(lang, LanguageConfig) else get_lang(lang)
    pattern = _build_tld_regex(cfg)
    m = pattern.search(url)
    if m:
        return cfg.country_for_tld(m.group(1).lower())
    return None


def iter_by_country(
    language: str | LanguageConfig = "ar",
    max_per_country: int = 100_000,
    min_text_length: int = 50,
    corpus: str = "mc4",
) -> Iterator[tuple[str, str]]:
    """Stream (country_iso, text) pairs from a multilingual web corpus.

    Requires: pip install datasets>=2.0

    Args:
        language:         Language code ("ar", "es", "pt", "fr") or LanguageConfig.
        max_per_country:  Cap per country to avoid dominant countries swamping others.
        min_text_length:  Skip documents shorter than this (characters).
        corpus:           "mc4" (default, public) or "oscar" (requires HF login).

    Yields:
        (country_iso, text) tuples.

    Note:
        mc4 (allenai/c4 multilingual) is publicly accessible and has URL metadata.
        OSCAR-2201 requires accepting the dataset license on huggingface.co and
        running `huggingface-cli login` first.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets>=2.0 required.") from e

    cfg = language if isinstance(language, LanguageConfig) else get_lang(language)
    pattern = _build_tld_regex(cfg)
    n_countries = len(set(cfg.tld_map.values()))

    counts: dict[str, int] = {}
    done: set[str] = set()

    if corpus == "oscar":
        ds = load_dataset(
            "oscar-corpus/OSCAR-2201",
            cfg.oscar_code,
            split="train",
            streaming=True,
        )
    else:
        # mc4: publicly available, has 'url' and 'text' fields
        ds = load_dataset(
            "allenai/c4",
            cfg.oscar_code,
            split="train",
            streaming=True,
        )

    for row in ds:
        if len(done) >= n_countries:
            break

        # mc4 has a top-level 'url' field; OSCAR nests it in metadata
        url = (
            row.get("url", "")
            or (row.get("meta") or {}).get("warc_headers", {}).get("warc-target-uri", "")
            or (row.get("meta") or {}).get("url", "")
        )

        m = pattern.search(url)
        if not m:
            continue
        country = cfg.country_for_tld(m.group(1).lower())
        if country is None or country in done:
            continue

        text: str = row.get("text", "")
        if len(text) < min_text_length:
            continue

        yield country, text
        counts[country] = counts.get(country, 0) + 1
        if counts[country] >= max_per_country:
            done.add(country)


def write_to_dir(
    out_dir: str | Path,
    language: str | LanguageConfig = "ar",
    max_per_country: int = 100_000,
    min_text_length: int = 50,
    corpus: str = "mc4",
) -> dict[str, int]:
    """Collect TLD-filtered text and write one file per country.

    Returns a {country: n_documents} summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, object] = {}
    counts: dict[str, int] = {}

    try:
        for country, text in iter_by_country(language, max_per_country, min_text_length, corpus=corpus):
            if country not in handles:
                handles[country] = (out_dir / f"{country}.txt").open("w", encoding="utf-8")
                counts[country] = 0
            handles[country].write(text.replace("\n", " ") + "\n")
            counts[country] += 1
    finally:
        for fh in handles.values():
            fh.close()

    return counts


def load_from_dir(root: str | Path) -> dict[str, list[str]]:
    """Load pre-written per-country text files (one document per line).

    Use this after write_to_dir() to avoid re-streaming from HuggingFace.
    """
    root = Path(root)
    out: dict[str, list[str]] = {}
    for p in sorted(root.glob("*.txt")):
        country = p.stem.upper()
        with p.open(encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        if lines:
            out[country] = lines
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Filter OSCAR 22.01 by country TLD.")
    ap.add_argument("--language", default="ar",
                    help="Language code: ar, es, pt, fr (default: ar)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for per-country text files.")
    ap.add_argument("--max-per-country", type=int, default=100_000)
    ap.add_argument("--min-text-length", type=int, default=50)
    ap.add_argument("--corpus", default="mc4", choices=["mc4", "oscar"],
                    help="mc4 (public, default) or oscar (requires HF login)")
    args = ap.parse_args()

    from data.geo_langs import get as get_lang
    cfg = get_lang(args.language)
    print(f"Language: {cfg.name}  |  OSCAR code: {cfg.oscar_code}")
    print(f"TLD coverage: {len(cfg.tld_map)} TLDs → {len(cfg.countries())} countries")
    print(f"Streaming OSCAR 22.01 {cfg.oscar_code} → {args.out_dir} ...")

    counts = write_to_dir(
        args.out_dir, cfg,
        args.max_per_country, args.min_text_length,
        corpus=args.corpus,
    )
    print("\nDocuments collected per country:")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {n:,}")
    print(f"\nTotal: {sum(counts.values()):,} docs across {len(counts)} countries.")
