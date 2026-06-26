"""Annotation cost curve: supervised macro-F1 on NADI dev vs. N labeled tweets.

Shows how many labeled examples a supervised n-gram model needs to match
the zero-shot geographic interpolation baseline (0.099 macro-F1).

Usage:
  python -m analysis.annotation_curve --nadi-dir datasets/nadi2020/NADI2020_shared_task/NADI_release
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM
from save_load import load as load_bundle


GEO_MACRO_F1  = 0.121   # geo macro-F1 over the 17 scored countries (excl. Bahrain)
ORACLE_MACRO_F1 = 0.193  # full 18-country oracle on NADI dev (excl. Bahrain)


def train_hierarchy(docs: dict[str, list[str]], order: int = 4) -> Hierarchy:
    hier = Hierarchy()
    hier.log_language_prior = {"ar": 0.0}
    counts: dict = {}
    for country, texts in docs.items():
        lm = NGramLM(order=order).fit(texts)
        cell = Cell(language="ar", region=country, lm=lm)
        hier.add_cell(cell)
        counts[("ar", country)] = len(texts)
    hier.fit_priors_from_counts(counts)
    return hier


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def macro_f1(gold: list[str], pred: list[str], countries: list[str]) -> float:
    from collections import Counter
    tp = Counter(); fp = Counter(); fn = Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    scores = [_f1(tp[c], fp[c], fn[c]) for c in countries]
    return sum(scores) / len(scores)


def predict(hier: Hierarchy, texts: list[str]) -> list[str]:
    return [hier.predict_dialect(t)[0][1] for t in texts]


def run(nadi_dir: Path, n_trials: int = 3, seed: int = 42) -> dict[int, float]:
    from data.nadi import load_split
    rng = random.Random(seed)

    train_examples = [e for e in load_split(nadi_dir, "train") if e.country]
    dev_examples   = [e for e in load_split(nadi_dir, "dev")   if e.country]

    dev_texts   = [e.text    for e in dev_examples]
    dev_gold    = [e.country for e in dev_examples]
    dev_countries = sorted(set(dev_gold))

    # Group training data by country
    train_by_country: dict[str, list[str]] = {}
    for e in train_examples:
        train_by_country.setdefault(e.country, []).append(e.text)

    # Countries with enough data for all N values
    n_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    max_n = max(n_values)
    eligible = {c: txts for c, txts in train_by_country.items()
                if len(txts) >= max_n and c in set(dev_gold)}

    print(f"  {len(eligible)} countries with >= {max_n} training examples")
    print(f"  dev: {len(dev_examples)} examples, {len(dev_countries)} countries")

    results: dict[int, list[float]] = {n: [] for n in n_values}
    results["all"] = []

    for trial in range(n_trials):
        for n in n_values:
            docs = {c: rng.sample(txts, n) for c, txts in eligible.items()}
            hier = train_hierarchy(docs)
            pred = predict(hier, dev_texts)
            f1 = macro_f1(dev_gold, pred,
                          [c for c in dev_countries if c in eligible])
            results[n].append(f1)
            print(f"  trial={trial+1}  N={n:>5}  macro-F1={f1:.3f}")

        # All available training data
        docs_all = {c: txts for c, txts in eligible.items()}
        hier_all = train_hierarchy(docs_all)
        pred_all = predict(hier_all, dev_texts)
        f1_all = macro_f1(dev_gold, pred_all,
                          [c for c in dev_countries if c in eligible])
        results["all"].append(f1_all)
        print(f"  trial={trial+1}  N=  all  macro-F1={f1_all:.3f}")

    return {k: sum(v) / len(v) for k, v in results.items()}


def plot(results: dict, out_path: Path) -> None:
    n_vals = sorted(k for k in results if isinstance(k, int))
    f1_vals = [results[n] for n in n_vals]
    f1_all = results["all"]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    ax.plot(n_vals, f1_vals, "o-", color="#2166ac", linewidth=1.8,
            markersize=5, label="Supervised (N tweets/country)")
    ax.axhline(GEO_MACRO_F1,  color="#d6604d", linewidth=1.5,
               linestyle="--", label=f"Geo interpolation ({GEO_MACRO_F1:.3f})")
    ax.axhline(ORACLE_MACRO_F1, color="#4dac26", linewidth=1.5,
               linestyle=":", label=f"Oracle ({ORACLE_MACRO_F1:.3f})")
    ax.axhline(f1_all, color="#2166ac", linewidth=1.0,
               linestyle="-.", alpha=0.6, label=f"Supervised all data ({f1_all:.3f})")

    # Crossover annotation
    for i in range(len(n_vals) - 1):
        if f1_vals[i] < GEO_MACRO_F1 <= f1_vals[i + 1]:
            ax.axvspan(n_vals[i], n_vals[i + 1], alpha=0.12, color="#d6604d")
            ax.annotate("crossover", xy=((n_vals[i] + n_vals[i + 1]) / 2, GEO_MACRO_F1),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=7, color="#d6604d",
                        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Labeled tweets per country (N)", fontsize=9)
    ax.set_ylabel("Macro-F1 (NADI dev)", fontsize=9)
    ax.set_title("Annotation cost vs.~zero-shot geographic interpolation", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nadi-dir", type=Path,
                    default=ROOT / "datasets/nadi2020/NADI2020_shared_task/NADI_release")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_path = ROOT / "paper2" / "figures" / "annotation_curve.pdf"
    print("Running annotation cost curve ...")
    results = run(args.nadi_dir, n_trials=args.trials, seed=args.seed)
    plot(results, out_path)
    print("\nResults:")
    for k, v in sorted(results.items(), key=lambda x: x[0] if isinstance(x[0], int) else 9999):
        print(f"  N={str(k):>6}  macro-F1={v:.3f}")


if __name__ == "__main__":
    main()
