"""Train a (language, country) hierarchy from geographically-labeled text.

Supports Arabic, Spanish, Portuguese, and French. Select via --language.

--- Training sources (--source) ---

  aratweet   Arap-Tweet per-country Arabic Twitter files (Arabic only)
             Requires: --data-dir pointing at Arap-Tweet directory.

  aratweet_hf  Abdelrahman-Rezk/Arabic_Dialect_Identification from HuggingFace
             440K tweets, 18 countries, geo-labeled (no dialect annotation).
             Use --stream to download live, or --data-dir for cached files.

  cc_tld     OSCAR-22.01 streamed and filtered by country ccTLD.
             Works for any supported language.
             Use --stream to fetch live, or --data-dir for pre-written files.

  nadi       NADI train split gold labels (Arabic only, oracle upper bound)
             Requires: --data-dir pointing at the NADI split directory.

  dslcc      DSLCC train split (Spanish, Portuguese, French, German)
             Requires: --data-dir pointing at the DSLCC directory.

--- Usage examples ---

  # Arabic oracle (NADI gold labels):
  python train_geo.py --language ar --source nadi \
      --data-dir datasets/nadi2020/NADI2020_shared_task/NADI_release \
      --out models/ar_oracle.pkl

  # Arabic weak supervision (HF Twitter, download live):
  python train_geo.py --language ar --source aratweet_hf --stream \
      --out models/ar_geo_twitter.pkl

  # Arabic weak supervision (HF Twitter, from cached files):
  python train_geo.py --language ar --source aratweet_hf \
      --data-dir datasets/arabic_twitter_hf --out models/ar_geo_twitter.pkl

  # Arabic weak supervision (original Arap-Tweet files):
  python train_geo.py --language ar --source aratweet \
      --data-dir datasets/aratweet --out models/ar_aratweet.pkl

  # Spanish CC-TLD (stream live):
  python train_geo.py --language es --source cc_tld --stream \
      --out models/es_cctld.pkl

  # Spanish from DSLCC (oracle for Spanish):
  python train_geo.py --language es --source dslcc \
      --data-dir datasets/dslcc --out models/es_oracle.pkl

  # Portuguese CC-TLD (pre-written):
  python train_geo.py --language pt --source cc_tld \
      --data-dir datasets/cc_pt --out models/pt_cctld.pkl

  # With bootstrap for low-resource cells:
  python train_geo.py --language ar --source aratweet \
      --data-dir datasets/aratweet --bootstrap \
      --unlabeled-dir datasets/cc_ar --out models/ar_aratweet_bootstrap.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bootstrap import BootstrapConfig, bootstrap_cell
from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM
from save_load import save
from data.geo_langs import get as get_lang


def _build_hierarchy(
    docs_by_country: dict[str, list[str]],
    language_tag: str,
    order: int,
    min_docs: int,
) -> Hierarchy:
    """Train one NGramLM per country and assemble a Hierarchy."""
    hier = Hierarchy()
    counts: dict[tuple[str, str | None], int] = {}
    for country, texts in docs_by_country.items():
        if len(texts) < min_docs:
            print(f"  skip {country}: only {len(texts)} docs (< {min_docs})")
            continue
        lm = NGramLM(order=order).fit(texts)
        hier.add_cell(Cell(language=language_tag, region=country, lm=lm))
        counts[(language_tag, country)] = len(texts)
        print(f"  trained {country}: {len(texts):,} docs")
    hier.fit_priors_from_counts(counts)
    return hier


def _load_unlabeled_pool(unlabeled_dir: Path) -> list[str]:
    pool: list[str] = []
    for p in sorted(unlabeled_dir.glob("*.txt")):
        with p.open(encoding="utf-8") as f:
            pool.extend(line.rstrip("\n") for line in f if line.strip())
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train geo-n-gram dialect hierarchy.")
    ap.add_argument("--language", default="ar",
                    help="Language code: ar, es, pt, fr (default: ar)")
    ap.add_argument("--source",
                    choices=["aratweet", "aratweet_hf", "cc_tld", "nadi", "dslcc", "dsltl"],
                    required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--stream", action="store_true",
                    help="Stream CC-TLD data live from HuggingFace.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-docs", type=int, default=10)
    ap.add_argument("--max-per-country", type=int, default=100_000)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--bootstrap-seed", type=int, default=500)
    ap.add_argument("--unlabeled-dir", type=Path, default=None)
    ap.add_argument("--bootstrap-threshold", type=float, default=0.80)
    args = ap.parse_args(argv)

    lang_cfg = get_lang(args.language)
    print(f"Language: {lang_cfg.name} ({args.language})  |  source: {args.source}")

    docs_by_country: dict[str, list[str]] = {}

    # ── Load training data ───────────────────────────────────────────────────

    if args.source == "aratweet":
        if args.language != "ar":
            ap.error("--source aratweet is Arabic-only; use --language ar")
        if args.data_dir is None:
            ap.error("--data-dir required for --source aratweet")
        from data.aratweet import load, documents_by_country as dbc
        print(f"Loading Arap-Tweet from {args.data_dir} ...")
        examples = load(args.data_dir)
        docs_by_country = dbc(examples)
        print(f"  {len(examples):,} tweets across {len(docs_by_country)} countries")

    elif args.source == "aratweet_hf":
        if args.language != "ar":
            ap.error("--source aratweet_hf is Arabic-only; use --language ar")
        from data.arabic_twitter_hf import documents_by_country as dbc_hf
        if args.stream or args.data_dir is None:
            from data.arabic_twitter_hf import load_hf
            print("Downloading Abdelrahman-Rezk/Arabic_Dialect_Identification from HuggingFace ...")
            hf_examples = load_hf("train", max_per_country=args.max_per_country)
            docs_by_country = dbc_hf(hf_examples)
            print(f"  {len(hf_examples):,} tweets across {len(docs_by_country)} countries")
        else:
            from data.arabic_twitter_hf import load_from_cache
            print(f"Loading cached Arabic Twitter HF data from {args.data_dir} ...")
            hf_examples = load_from_cache(args.data_dir)
            docs_by_country = dbc_hf(hf_examples)
            print(f"  {len(hf_examples):,} tweets across {len(docs_by_country)} countries")

    elif args.source == "nadi":
        if args.language != "ar":
            ap.error("--source nadi is Arabic-only; use --language ar")
        if args.data_dir is None:
            ap.error("--data-dir required for --source nadi")
        from data.nadi import load_split, documents_by_country as dbc
        print(f"Loading NADI train from {args.data_dir} ...")
        examples = load_split(args.data_dir, "train")
        docs_by_country = dbc(examples)
        print(f"  {len(examples):,} tweets across {len(docs_by_country)} countries")

    elif args.source == "dslcc":
        if args.data_dir is None:
            ap.error("--data-dir required for --source dslcc")
        from data.dsl import load_dslcc, documents_by_country_dslcc
        print(f"Loading DSLCC train from {args.data_dir} ...")
        examples = load_dslcc(args.data_dir, "train", language=args.language)
        docs_by_country = documents_by_country_dslcc(examples)
        print(f"  {len(examples):,} examples across {len(docs_by_country)} varieties: "
              f"{sorted(docs_by_country)}")

    elif args.source == "dsltl":
        if args.data_dir is None:
            ap.error("--data-dir required for --source dsltl")
        from data.dsl import load_dsltl, documents_by_country_dslcc
        print(f"Loading DSL-TL {args.language} train from {args.data_dir} ...")
        examples = load_dsltl(args.data_dir, "train", language=args.language)
        docs_by_country = documents_by_country_dslcc(examples)
        print(f"  {len(examples):,} examples across {len(docs_by_country)} varieties: "
              f"{sorted(docs_by_country)}")

    elif args.source == "cc_tld":
        if args.stream:
            print(f"Streaming OSCAR 22.01 {args.language} (may take hours) ...")
            from data.cc_tld import iter_by_country
            for country, text in iter_by_country(args.language, args.max_per_country):
                docs_by_country.setdefault(country, []).append(text)
            total = sum(len(v) for v in docs_by_country.values())
            print(f"  collected {total:,} docs across {len(docs_by_country)} countries")
        elif args.data_dir is not None:
            from data.cc_tld import load_from_dir
            print(f"Loading CC-TLD files from {args.data_dir} ...")
            docs_by_country = load_from_dir(args.data_dir)
            total = sum(len(v) for v in docs_by_country.values())
            print(f"  {total:,} docs across {len(docs_by_country)} countries")
        else:
            ap.error("--source cc_tld requires either --stream or --data-dir")

    # ── Train ────────────────────────────────────────────────────────────────

    print(f"\nTraining n-gram LMs (order={args.order}) ...")
    hier = _build_hierarchy(docs_by_country, args.language, args.order, args.min_docs)
    print(f"  built {len(hier.cells)} cells")

    # ── Bootstrap low-resource cells ─────────────────────────────────────────

    if args.bootstrap:
        low_resource = {
            c: texts for c, texts in docs_by_country.items()
            if 0 < len(texts) < args.bootstrap_seed
        }
        if not low_resource:
            print("Bootstrap: no cells below seed threshold.")
        else:
            pool: list[str] = []
            if args.unlabeled_dir:
                pool = _load_unlabeled_pool(args.unlabeled_dir)
                print(f"Bootstrap pool: {len(pool):,} lines from {args.unlabeled_dir}")
            else:
                for texts in docs_by_country.values():
                    pool.extend(texts)

            cfg = BootstrapConfig(
                order=args.order,
                confidence_threshold=args.bootstrap_threshold,
            )
            for country, texts in low_resource.items():
                print(f"  bootstrapping {country} (seed={len(texts)}) ...")
                cell, delta = bootstrap_cell(
                    language=args.language,
                    region=country,
                    seed_texts=texts,
                    unlabeled_pool=iter(pool),
                    hierarchy=hier,
                    config=cfg,
                )
                hier.cells = [
                    c for c in hier.cells
                    if not (c.region == country and c.language == args.language)
                ]
                hier.add_cell(cell)
                print(f"    delta={delta:+.3f} ({'expanded' if delta >= 0 else 'seed-only'})")

    # ── Save ─────────────────────────────────────────────────────────────────

    save(
        hier, args.out,
        language=args.language,
        source=args.source,
        order=args.order,
        countries=sorted({c.region for c in hier.cells}),
    )
    print(f"\nSaved → {args.out}")
    print(f"  cells: {len(hier.cells)}  |  countries: {len({c.region for c in hier.cells})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
