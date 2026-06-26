"""Leave-one-out evaluation using CC-TLD data directly.

For each language (Spanish 19 countries, Portuguese 3 countries):
  - Split each country's docs 80% train / 20% test
  - Hold out one country at a time
  - Oracle: train on all 80% splits including held-out country
  - Geo:    train on all 80% splits EXCEPT held-out; replace with mixture
  - Evaluate: per-country F1 on the held-out country's test docs
              (full N-class classifier over all test docs)

Docs are truncated to MAX_CHARS characters; test capped at MAX_TEST per country.
Run time: ~5 minutes for Spanish, <1 minute for Portuguese.

Usage:
  python -m analysis.cctld_loo [--lang es|pt|both]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM
from geo_interpolate import CENTROIDS, geographic_weights, MixtureNGramLM

MAX_CHARS = 500   # truncate long web docs
MAX_TEST  = 200   # max test docs per country
MIN_DOCS  = 80    # skip countries with fewer docs

LANG_DIRS = {
    "es": ROOT / "datasets" / "cc_es",
    "pt": ROOT / "datasets" / "cc_pt",
    "fr": ROOT / "datasets" / "cc_fr",
    "de": ROOT / "datasets" / "cc_de",
    "en": ROOT / "datasets" / "cc_en",
    "ru": ROOT / "datasets" / "cc_ru",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_country_docs(data_dir: Path, min_docs: int = MIN_DOCS) -> dict[str, list[str]]:
    """Load {country: [doc, ...]} from a CC-TLD directory of per-country .txt files."""
    docs: dict[str, list[str]] = {}
    for f in sorted(data_dir.glob("*.txt")):
        country = f.stem.upper()
        if country not in CENTROIDS:
            continue
        lines = [ln.strip()[:MAX_CHARS] for ln in f.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        if len(lines) >= min_docs:
            docs[country] = lines
    return docs


def train_test_split(
    docs: dict[str, list[str]],
    test_frac: float = 0.20,
    max_test: int = MAX_TEST,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rng = random.Random(seed)
    train, test = {}, {}
    for country, lines in docs.items():
        shuffled = lines[:]
        rng.shuffle(shuffled)
        n_test = min(max(1, int(len(shuffled) * test_frac)), max_test)
        test[country]  = shuffled[:n_test]
        train[country] = shuffled[n_test:]
    return train, test


# ── Model training ────────────────────────────────────────────────────────────

def build_hierarchy(
    train_docs: dict[str, list[str]],
    language: str,
    order: int = 4,
) -> Hierarchy:
    hier = Hierarchy()
    hier.log_language_prior = {language: 0.0}
    counts: dict = {}
    for country, texts in train_docs.items():
        lm = NGramLM(order=order).fit(texts)
        cell = Cell(language=language, region=country, lm=lm)
        hier.add_cell(cell)
        counts[(language, country)] = len(texts)
    hier.fit_priors_from_counts(counts)
    return hier


# ── LOO evaluation ─────────────────────────────────────────────────────────────

def _f1(gold: list[str], pred: list[str], country: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == country and p == country)
    fp = sum(1 for g, p in zip(gold, pred) if g != country and p == country)
    fn = sum(1 for g, p in zip(gold, pred) if g == country and p != country)
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-12)


def _cache_lp(cells: list[Cell], texts: list[str]) -> dict[str, list[float]]:
    return {c.region: [c.lm.logprob(t) for t in texts]
            for c in cells if c.region}


def _predict_with_cache(
    lp_cache: dict[str, list[float]],
    cell_by_region: dict[str, Cell],
    lang_prior: float,
    held_out: str,
    mix_lm: MixtureNGramLM,
    mix_prior: float,
    texts: list[str],
) -> list[str]:
    known = [r for r in lp_cache if r != held_out]
    preds = []
    for i, text in enumerate(texts):
        scores: dict[str, float] = {}
        for r in known:
            c = cell_by_region[r]
            scores[r] = lp_cache[r][i] + c.log_prior + lang_prior
        scores[held_out] = mix_lm.logprob(text) + mix_prior + lang_prior
        preds.append(max(scores, key=scores.__getitem__))
    return preds


def loo_evaluate_lang(
    lang: str,
    data_dir: Path,
    k: int = 5,
) -> dict[str, dict]:
    docs = load_country_docs(data_dir)
    countries = sorted(docs)
    print(f"\n{lang.upper()}: {len(countries)} countries  "
          f"(docs: {', '.join(f'{c}={len(docs[c])}' for c in countries)})",
          flush=True)

    train_docs, test_docs = train_test_split(docs)
    print(f"  train: {sum(len(v) for v in train_docs.values()):,}  "
          f"test: {sum(len(v) for v in test_docs.values()):,}", flush=True)

    # Flatten test set: texts + gold labels
    test_texts, test_gold = [], []
    for country, txts in sorted(test_docs.items()):
        test_texts.extend(txts)
        test_gold.extend([country] * len(txts))

    # Oracle: full model (all countries trained)
    print("  Training oracle ...", end=" ", flush=True)
    oracle_hier = build_hierarchy(train_docs, lang)
    print("done. Pre-computing oracle logprobs ...", end=" ", flush=True)
    oracle_cells = oracle_hier.cells
    lp_oracle = _cache_lp(oracle_cells, test_texts)
    oracle_lang_prior = oracle_hier.log_language_prior.get(lang, 0.0)
    oracle_cell_map = {c.region: c for c in oracle_cells}
    print("done", flush=True)

    results: dict[str, dict] = {}

    for held_out in countries:
        # Oracle F1 for this country
        pred_oracle = []
        for i in range(len(test_texts)):
            scores = {r: lp_oracle[r][i] + oracle_cell_map[r].log_prior + oracle_lang_prior
                      for r in lp_oracle}
            pred_oracle.append(max(scores, key=scores.__getitem__))
        f1_oracle = _f1(test_gold, pred_oracle, held_out)

        # Geo interpolation: train without held_out, replace with mixture
        loo_train = {c: v for c, v in train_docs.items() if c != held_out}
        loo_hier = build_hierarchy(loo_train, lang)
        loo_cells = loo_hier.cells
        lang_prior = loo_hier.log_language_prior.get(lang, 0.0)

        # Pre-compute logprobs for loo cells (only known cells changed by removing held_out)
        lp_loo = _cache_lp(loo_cells, test_texts)
        cell_map = {c.region: c for c in loo_cells}

        known = list(cell_map.keys())
        w = geographic_weights(held_out, known, k=min(k, len(known)))
        src_cells = [cell_map[c] for c in w]
        mix = MixtureNGramLM([c.lm for c in src_cells], list(w.values()))
        avg_prior = sum(c.log_prior for c in src_cells) / len(src_cells)

        pred_geo = _predict_with_cache(
            lp_loo, cell_map, lang_prior, held_out, mix, avg_prior, test_texts)
        f1_geo = _f1(test_gold, pred_geo, held_out)

        nn = min((c for c in known if c in CENTROIDS),
                 key=lambda c: math.dist(
                     [CENTROIDS[held_out][0], CENTROIDS[held_out][1]],
                     [CENTROIDS[c][0],        CENTROIDS[c][1]]),
                 default="?")
        from geo_interpolate import haversine
        dist_nn = haversine(CENTROIDS[held_out], CENTROIDS[nn]) if nn != "?" else float("nan")

        results[held_out] = {
            "oracle_f1":  f1_oracle,
            "geo_f1":     f1_geo,
            "dist_nn_km": dist_nn,
            "nn_country": nn,
            "recovery":   f1_geo / max(f1_oracle, 1e-6),
            "n_test":     len(test_docs[held_out]),
        }
        print(
            f"  {held_out:<3}  oracle={f1_oracle:.3f}  geo={f1_geo:.3f}  "
            f"recovery={f1_geo/max(f1_oracle,1e-6):.1%}  "
            f"dist={dist_nn:>6.0f}km  nn={nn}",
            flush=True,
        )

    # Macro (exclude countries with oracle ≈ 0). Recovery is the ratio of macro
    # F1s (geo / oracle), matching the paper's definition, not the mean of
    # per-country ratios (which a near-zero oracle would dominate).
    valid = {c: r for c, r in results.items() if r["oracle_f1"] > 0.01}
    n = len(valid)
    if n:
        macro_oracle = sum(r["oracle_f1"] for r in valid.values()) / n
        macro_geo    = sum(r["geo_f1"]    for r in valid.values()) / n
        results["MACRO"] = {
            "oracle_f1":  macro_oracle,
            "geo_f1":     macro_geo,
            "recovery":   macro_geo / max(macro_oracle, 1e-9),
            "dist_nn_km": float("nan"),
            "nn_country": "---",
            "n_test":     sum(r["n_test"] for r in valid.values()),
        }
    return results


def print_results(results: dict[str, dict], lang: str) -> None:
    print(f"\n{'='*64}")
    print(f"{lang.upper()} CC-TLD Leave-One-Out")
    print(f"{'='*64}")
    hdr = f"{'country':<6}  {'oracle':>7}  {'geo':>7}  {'dist_km':>8}  {'recovery':>9}  nn"
    print(hdr)
    print("-" * len(hdr))
    for country, r in results.items():
        dist = f"{r['dist_nn_km']:>8.0f}" if not math.isnan(r["dist_nn_km"]) else f"{'---':>8}"
        rec  = f"{r['recovery']:>9.1%}" if isinstance(r["recovery"], float) else f"{'---':>9}"
        print(f"{country:<6}  {r['oracle_f1']:>7.3f}  {r['geo_f1']:>7.3f}  "
              f"{dist}  {rec}  {r['nn_country']}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["es", "pt", "fr", "de", "en", "ru", "both"],
                    default="both")
    ap.add_argument("--k",    type=int, default=5)
    args = ap.parse_args(argv)

    all_langs = ["es", "pt", "fr", "de", "en", "ru"]
    langs = all_langs if args.lang == "both" else [args.lang]

    all_results: dict[str, dict] = {}
    for lang in langs:
        data_dir = LANG_DIRS[lang]
        if not data_dir.exists():
            print(f"\nSkipping {lang.upper()}: {data_dir} does not exist.", flush=True)
            continue
        results = loo_evaluate_lang(lang, data_dir, k=args.k)
        print_results(results, lang)
        all_results[lang] = results

    return all_results


if __name__ == "__main__":
    main()
