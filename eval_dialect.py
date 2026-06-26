"""Language-agnostic dialect/variety identification evaluation.

Evaluates geo-n-gram hierarchies against zero-shot baselines (fastText, GlotLID)
on any supported language. Works across:

  Arabic    NADI 2020 (21 countries)
  Spanish   DSLCC (3-5 varieties: AR, MX, ES, PE, CL)
  Portuguese DSLCC / DSL-TL (BR vs PT)

--- Metrics ---

  accuracy     % correct
  macro_F1     unweighted mean F1 across all varieties in the eval set
  covered_F1   macro-F1 restricted to varieties the model can actually predict
               (exposes how many zero-coverage varieties fastText/GlotLID have)

--- Usage ---

  # Arabic (NADI):
  python eval_dialect.py --language ar \
      --eval-source nadi --eval-dir datasets/nadi2020/NADI2020_shared_task/NADI_release \
      --models models/ar_oracle.pkl:oracle models/ar_aratweet.pkl:geo_twitter \
      --per-variety

  # Spanish (DSLCC):
  python eval_dialect.py --language es \
      --eval-source dslcc --eval-dir datasets/dslcc \
      --models models/es_oracle.pkl:oracle models/es_cctld.pkl:geo_cc \
      --per-variety

  # Portuguese (DSL-TL):
  python eval_dialect.py --language pt \
      --eval-source dsltl --eval-dir datasets/DSL-TL/PT \
      --models models/pt_oracle.pkl:oracle models/pt_cctld.pkl:geo_cc \
      --per-variety

  # Baselines only (no models needed):
  python eval_dialect.py --language ar \
      --eval-source nadi --eval-dir datasets/nadi2020/NADI2020_shared_task/NADI_release \
      --baselines-only
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from hierarchy import Hierarchy
from save_load import load as load_bundle


# ── Metrics ───────────────────────────────────────────────────────────────────

def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def compute_metrics(
    gold: list[str],
    pred: list[str | None],
    varieties: list[str],
) -> dict:
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

    per_variety: dict[str, float] = {v: _f1(tp[v], fp[v], fn[v]) for v in varieties}
    macro_f1 = sum(per_variety.values()) / max(len(per_variety), 1)

    predicted_set = {p for p in pred if p is not None}
    covered = [v for v in varieties if v in predicted_set]
    covered_f1 = (
        sum(per_variety[v] for v in covered) / len(covered) if covered else 0.0
    )

    return {
        "accuracy": n_correct / max(len(gold), 1),
        "macro_f1": macro_f1,
        "covered_f1": covered_f1,
        "n_covered": len(covered),
        "n_varieties": len(varieties),
        "per_variety": per_variety,
        "n_examples": len(gold),
        "n_abstain": sum(1 for p in pred if p is None),
    }


# ── Load eval data ────────────────────────────────────────────────────────────

def _load_eval_data(
    source: str, eval_dir: Path, language: str, split: str
) -> tuple[list[str], list[str]]:
    """Return (texts, gold_labels) for the eval split."""
    if source == "nadi":
        from data.nadi import load_split
        examples = load_split(eval_dir, split)
        examples = [e for e in examples if e.country]
        return [e.text for e in examples], [e.country for e in examples]

    elif source == "dslcc":
        from data.dsl import load_dslcc
        examples = load_dslcc(eval_dir, split, language=language)
        return [e.text for e in examples], [e.country for e in examples]

    elif source == "dsltl":
        from data.dsl import load_dsltl
        examples = load_dsltl(eval_dir, split, language=language)
        return [e.text for e in examples], [e.country for e in examples]

    elif source == "madar":
        from data.madar import load_split
        examples = load_split(eval_dir, split)
        examples = [e for e in examples if e.country != "MSA"]
        return [e.text for e in examples], [e.country for e in examples]

    else:
        raise ValueError(f"Unknown eval-source: {source!r}")


# ── Prediction ────────────────────────────────────────────────────────────────

def _predict_hier(hier: Hierarchy, texts: list[str]) -> list[str | None]:
    preds = []
    for text in texts:
        cell, _ = hier.predict_dialect(text)
        preds.append(cell[1])  # (language, region) → region
    return preds


def _predict_baseline(model, texts: list[str]) -> list[str | None]:
    return model.predict_batch(texts)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(results: dict[str, dict], varieties: list[str]) -> None:
    header = f"{'model':<22} {'acc':>7} {'macroF1':>9} {'covF1':>8} {'covered':>9}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<22} "
            f"{r['accuracy']:>7.3f} "
            f"{r['macro_f1']:>9.3f} "
            f"{r['covered_f1']:>8.3f} "
            f"{r['n_covered']:>4}/{r['n_varieties']}"
        )


def _print_per_variety(results: dict[str, dict], varieties: list[str]) -> None:
    col_w = 9
    header = f"{'variety':<8}" + "".join(f"{m[:col_w]:>{col_w}}" for m in results)
    print(header)
    print("-" * len(header))
    for v in varieties:
        row = f"{v:<8}"
        for r in results.values():
            row += f"{r['per_variety'].get(v, 0.0):>{col_w}.3f}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="ar",
                    help="Language code: ar, es, pt, fr (default: ar)")
    ap.add_argument("--eval-source",
                    choices=["nadi", "dslcc", "dsltl", "madar"],
                    required=True,
                    help="Evaluation benchmark to use.")
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Directory containing the evaluation data.")
    ap.add_argument("--split", default="dev",
                    help="Eval split: train/dev/test (default: dev)")
    ap.add_argument("--models", nargs="*", default=[],
                    metavar="PATH:NAME",
                    help="Trained model .pkl files with display name, e.g. "
                         "models/ar_oracle.pkl:oracle models/ar_aratweet.pkl:geo_twitter")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--no-fasttext", action="store_true")
    ap.add_argument("--no-glotlid", action="store_true")
    ap.add_argument("--per-variety", action="store_true")
    args = ap.parse_args(argv)

    # ── Load eval data ───────────────────────────────────────────────────────
    print(f"Loading {args.eval_source} {args.split} from {args.eval_dir} ...")
    texts, gold = _load_eval_data(args.eval_source, args.eval_dir, args.language, args.split)
    varieties = sorted(set(gold))
    print(f"  {len(texts):,} examples  |  {len(varieties)} varieties: {varieties}")

    results: dict[str, dict] = {}

    # ── Baselines ────────────────────────────────────────────────────────────
    if not args.no_fasttext:
        print("\nRunning fastText lid.218 ...")
        try:
            from baselines.fasttext_lid import FasttextLID
            ft = FasttextLID()
            results["fasttext"] = compute_metrics(gold, _predict_baseline(ft, texts), varieties)
            cov = ft.coverage(varieties)
            print(f"  fastText covers {cov['coverage_pct']:.0f}% of varieties")
            if cov["uncovered"]:
                print(f"  uncovered: {cov['uncovered']}")
        except Exception as e:
            print(f"  [skipped: {e}]")

    if not args.no_glotlid:
        print("\nRunning GlotLID v3 ...")
        try:
            from baselines.fasttext_lid import GlotLID
            glot = GlotLID()
            results["glotlid"] = compute_metrics(gold, _predict_baseline(glot, texts), varieties)
        except Exception as e:
            print(f"  [skipped: {e}]")

    if args.baselines_only:
        print()
        _print_summary(results, varieties)
        if args.per_variety:
            print()
            _print_per_variety(results, varieties)
        return 0

    # ── Trained models ───────────────────────────────────────────────────────
    for spec in args.models:
        if ":" in spec:
            path_str, name = spec.rsplit(":", 1)
        else:
            path_str, name = spec, Path(spec).stem
        path = Path(path_str)
        print(f"\nLoading model '{name}' from {path} ...")
        bundle = load_bundle(path)
        hier = bundle.hierarchy
        meta = bundle.metadata
        print(f"  cells: {len(hier.cells)}  |  source: {meta.get('source', '?')}  "
              f"|  order: {meta.get('order', '?')}")
        results[name] = compute_metrics(gold, _predict_hier(hier, texts), varieties)

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{args.language.upper()} | {args.eval_source} {args.split}  "
          f"({len(texts):,} examples, {len(varieties)} varieties)")
    print(f"{'='*65}")
    _print_summary(results, varieties)

    if args.per_variety and results:
        print()
        _print_per_variety(results, varieties)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
