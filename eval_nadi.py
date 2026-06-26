"""Evaluate geo-n-gram hierarchies and baselines on NADI country-level LID.

This is the main experiment script. It runs all conditions in one pass and
prints a comparison table.

--- Conditions ---

  geo_aratweet   Trained on Arap-Tweet geo-labeled tweets (weak supervision)
  geo_cctld      Trained on OSCAR CC-TLD filtered text (distant supervision)
  oracle         Trained on NADI train gold labels (upper bound)
  fasttext       fastText lid.218 zero-shot (baseline)
  glotlid        GlotLID v3 zero-shot (baseline)

--- Metrics ---

  accuracy       % of examples correctly predicted
  macro_F1       unweighted mean F1 across all countries in the eval set
  covered_F1     macro-F1 restricted to countries the model can predict
                 (relevant for fastText/GlotLID which have no Gulf coverage)
  per_country    F1 for each of the 21 NADI countries

--- Usage ---

  # Minimal: just run fastText and GlotLID baselines (no training needed)
  python eval_nadi.py --nadi-dir datasets/nadi2020 --baselines-only

  # Full experiment with a trained hierarchy:
  python eval_nadi.py \\
      --nadi-dir datasets/nadi2020 \\
      --aratweet-model models/hier_aratweet.pkl \\
      --cctld-model    models/hier_cctld.pkl \\
      --oracle-model   models/hier_nadi_oracle.pkl

  # MADAR cross-domain eval (requires MADAR data):
  python eval_nadi.py \\
      --nadi-dir datasets/nadi2020 \\
      --madar-dir datasets/madar \\
      --aratweet-model models/hier_aratweet.pkl

  # Train on-the-fly (no pre-trained model needed):
  python eval_nadi.py \\
      --nadi-dir datasets/nadi2020 \\
      --aratweet-dir datasets/aratweet
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from hierarchy import Hierarchy
from save_load import load as load_bundle


# ── Metrics ──────────────────────────────────────────────────────────────────

def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def compute_metrics(
    gold: list[str],
    pred: list[str | None],
    countries: list[str],
) -> dict:
    """Compute accuracy, macro-F1, covered-F1, and per-country F1.

    pred entries may be None (model abstains / returns no prediction).
    Abstentions count as wrong for accuracy and per-country recall.
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    n_correct = 0

    for g, p in zip(gold, pred):
        if p == g:
            n_correct += 1
            tp[g] += 1
        else:
            if p is not None:
                fp[p] += 1
            fn[g] += 1

    per_country: dict[str, float] = {}
    for c in countries:
        per_country[c] = _f1(tp[c], fp[c], fn[c])

    macro_f1 = sum(per_country.values()) / max(len(per_country), 1)

    # covered_F1: only countries for which the model made ≥1 non-None prediction
    predicted_countries = {p for p in pred if p is not None}
    covered = [c for c in countries if c in predicted_countries]
    covered_f1 = (
        sum(per_country[c] for c in covered) / len(covered) if covered else 0.0
    )

    return {
        "accuracy": n_correct / max(len(gold), 1),
        "macro_f1": macro_f1,
        "covered_f1": covered_f1,
        "n_covered": len(covered),
        "n_countries": len(countries),
        "per_country": per_country,
        "n_examples": len(gold),
        "n_abstain": sum(1 for p in pred if p is None),
    }


# ── Prediction helpers ────────────────────────────────────────────────────────

def _predict_hierarchy(hier: Hierarchy, texts: list[str]) -> list[str | None]:
    preds = []
    for text in texts:
        cell, _ = hier.predict_dialect(text)
        preds.append(cell[1])  # cell is (language, region)
    return preds


def _predict_baseline(model, texts: list[str]) -> list[str | None]:
    return model.predict_batch(texts)


# ── Training on-the-fly ───────────────────────────────────────────────────────

def _train_from_aratweet(data_dir: Path, order: int) -> Hierarchy:
    from data.aratweet import load, documents_by_country
    from hierarchy import Cell
    from ngram_lm import NGramLM

    print(f"  Loading Arap-Tweet from {data_dir} ...")
    examples = load(data_dir)
    docs = documents_by_country(examples)
    hier = Hierarchy()
    counts: dict = {}
    for country, texts in docs.items():
        lm = NGramLM(order=order).fit(texts)
        hier.add_cell(Cell(language="ar", region=country, lm=lm))
        counts[("ar", country)] = len(texts)
    hier.fit_priors_from_counts(counts)
    print(f"  Built {len(hier.cells)} cells from {sum(len(v) for v in docs.values()):,} tweets")
    return hier


def _train_oracle(data_dir: Path, order: int) -> Hierarchy:
    from data.nadi import load_split, documents_by_country
    from hierarchy import Cell
    from ngram_lm import NGramLM

    print(f"  Loading NADI train from {data_dir} ...")
    examples = load_split(data_dir, "train")
    docs = documents_by_country(examples)
    hier = Hierarchy()
    counts: dict = {}
    for country, texts in docs.items():
        lm = NGramLM(order=order).fit(texts)
        hier.add_cell(Cell(language="ar", region=country, lm=lm))
        counts[("ar", country)] = len(texts)
    hier.fit_priors_from_counts(counts)
    print(f"  Built {len(hier.cells)} cells from {sum(len(v) for v in docs.values()):,} tweets")
    return hier


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_table(results: dict[str, dict], countries: list[str]) -> None:
    header = f"{'model':<20} {'acc':>7} {'macroF1':>9} {'covF1':>8} {'covered':>9}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<20} "
            f"{r['accuracy']:>7.3f} "
            f"{r['macro_f1']:>9.3f} "
            f"{r['covered_f1']:>8.3f} "
            f"{r['n_covered']:>4}/{r['n_countries']}"
        )


def _print_per_country(results: dict[str, dict], countries: list[str]) -> None:
    col_w = 8
    header = f"{'country':<6}" + "".join(f"{m[:col_w]:>{col_w}}" for m in results)
    print(header)
    print("-" * len(header))
    for c in countries:
        row = f"{c:<6}"
        for r in results.values():
            f1 = r["per_country"].get(c, 0.0)
            row += f"{f1:>{col_w}.3f}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nadi-dir", type=Path, required=True,
                    help="Directory with NADI TSV files.")
    ap.add_argument("--nadi-split", default="dev",
                    help="Which NADI split to evaluate on (default: dev).")
    ap.add_argument("--madar-dir", type=Path, default=None,
                    help="Optional: also evaluate on MADAR (cross-domain).")

    # Pre-trained model paths
    ap.add_argument("--aratweet-model", type=Path, default=None,
                    help="Path to hier_aratweet.pkl from train_geo.py")
    ap.add_argument("--cctld-model", type=Path, default=None,
                    help="Path to hier_cctld.pkl from train_geo.py")
    ap.add_argument("--oracle-model", type=Path, default=None,
                    help="Path to hier_nadi_oracle.pkl from train_geo.py")

    # On-the-fly training shortcuts
    ap.add_argument("--aratweet-dir", type=Path, default=None,
                    help="Train on Arap-Tweet on the fly (no .pkl needed).")
    ap.add_argument("--oracle-train", action="store_true",
                    help="Train oracle on-the-fly using NADI train split.")

    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--baselines-only", action="store_true",
                    help="Run only fastText + GlotLID, skip hierarchy models.")
    ap.add_argument("--no-fasttext", action="store_true")
    ap.add_argument("--no-glotlid", action="store_true")
    ap.add_argument("--per-country", action="store_true",
                    help="Print per-country F1 table.")
    args = ap.parse_args(argv)

    # --- Load eval data ---
    from data.nadi import load_split, label_set
    print(f"Loading NADI {args.nadi_split} from {args.nadi_dir} ...")
    eval_examples = load_split(args.nadi_dir, args.nadi_split)
    eval_examples = [e for e in eval_examples if e.country]
    texts = [e.text for e in eval_examples]
    gold = [e.country for e in eval_examples]
    countries = label_set(eval_examples)
    print(f"  {len(eval_examples):,} examples  |  {len(countries)} countries")

    results: dict[str, dict] = {}

    # --- Baseline: fastText ---
    if not args.no_fasttext:
        print("\nRunning fastText lid.218 ...")
        try:
            from baselines.fasttext_lid import FasttextLID
            ft = FasttextLID()
            pred_ft = _predict_baseline(ft, texts)
            results["fasttext"] = compute_metrics(gold, pred_ft, countries)
            cov = ft.coverage(countries)
            print(f"  fastText covers {cov['coverage_pct']:.0f}% of eval countries")
            if cov["uncovered"]:
                print(f"  uncovered: {cov['uncovered']}")
        except Exception as e:
            print(f"  [skipped: {e}]")

    # --- Baseline: GlotLID ---
    if not args.no_glotlid:
        print("\nRunning GlotLID v3 ...")
        try:
            from baselines.fasttext_lid import GlotLID
            glot = GlotLID()
            pred_glot = _predict_baseline(glot, texts)
            results["glotlid"] = compute_metrics(gold, pred_glot, countries)
        except Exception as e:
            print(f"  [skipped: {e}]")

    if args.baselines_only:
        print()
        _print_table(results, countries)
        if args.per_country:
            print()
            _print_per_country(results, countries)
        return 0

    # --- Geo hierarchy: Arap-Tweet ---
    if args.aratweet_model or args.aratweet_dir:
        print("\nLoading/training geo_aratweet hierarchy ...")
        if args.aratweet_model:
            bundle = load_bundle(args.aratweet_model)
            hier_at = bundle.hierarchy
            print(f"  loaded {len(hier_at.cells)} cells from {args.aratweet_model}")
        else:
            hier_at = _train_from_aratweet(args.aratweet_dir, args.order)
        pred_at = _predict_hierarchy(hier_at, texts)
        results["geo_aratweet"] = compute_metrics(gold, pred_at, countries)

    # --- Geo hierarchy: CC-TLD ---
    if args.cctld_model:
        print("\nLoading geo_cctld hierarchy ...")
        bundle = load_bundle(args.cctld_model)
        hier_cc = bundle.hierarchy
        print(f"  loaded {len(hier_cc.cells)} cells from {args.cctld_model}")
        pred_cc = _predict_hierarchy(hier_cc, texts)
        results["geo_cctld"] = compute_metrics(gold, pred_cc, countries)

    # --- Oracle: NADI gold labels ---
    if args.oracle_model or args.oracle_train:
        print("\nLoading/training oracle hierarchy ...")
        if args.oracle_model:
            bundle = load_bundle(args.oracle_model)
            hier_or = bundle.hierarchy
            print(f"  loaded {len(hier_or.cells)} cells from {args.oracle_model}")
        else:
            hier_or = _train_oracle(args.nadi_dir, args.order)
        pred_or = _predict_hierarchy(hier_or, texts)
        results["oracle"] = compute_metrics(gold, pred_or, countries)

    # --- MADAR cross-domain eval ---
    trained_hiers: dict[str, Hierarchy] = {}
    if "geo_aratweet" in results and args.aratweet_model:
        trained_hiers["geo_aratweet"] = load_bundle(args.aratweet_model).hierarchy
    elif "geo_aratweet" in results:
        trained_hiers["geo_aratweet"] = hier_at  # type: ignore[possibly-undefined]
    if "geo_cctld" in results:
        trained_hiers["geo_cctld"] = load_bundle(args.cctld_model).hierarchy
    if "oracle" in results and args.oracle_model:
        trained_hiers["oracle"] = load_bundle(args.oracle_model).hierarchy
    elif "oracle" in results:
        trained_hiers["oracle"] = hier_or  # type: ignore[possibly-undefined]

    if args.madar_dir:
        print("\n--- MADAR cross-domain evaluation (country-level) ---")
        from data.madar import load_split as load_madar
        madar_examples = load_madar(args.madar_dir, "dev")
        madar_texts = [e.text for e in madar_examples if e.country != "MSA"]
        madar_gold = [e.country for e in madar_examples if e.country != "MSA"]
        madar_countries = sorted(set(madar_gold))
        print(f"  {len(madar_texts):,} MADAR examples  |  {len(madar_countries)} countries")

        madar_results: dict[str, dict] = {}
        for name, hier in trained_hiers.items():
            pred = _predict_hierarchy(hier, madar_texts)
            madar_results[name] = compute_metrics(madar_gold, pred, madar_countries)

        print("\n  MADAR results:")
        _print_table(madar_results, madar_countries)

    # --- Print NADI results ---
    print(f"\n{'='*60}")
    print(f"NADI {args.nadi_split} results  ({len(eval_examples):,} examples, {len(countries)} countries)")
    print(f"{'='*60}")
    _print_table(results, countries)

    if args.per_country and results:
        print()
        _print_per_country(results, countries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
