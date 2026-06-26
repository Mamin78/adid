"""Zero-shot Arabic dialect synthesis via geographic n-gram interpolation.

For a held-out country, blends n-gram LMs from the K geographically nearest
known countries (weighted by 1/distance), enabling prediction on dialects
absent from training data.

Leave-one-out evaluation compares four conditions per country:
  oracle    trained on real data for that country (upper bound)
  geo       K-nearest-neighbor mixture, inverse-distance weighted
  nearest   single nearest neighbor only
  uniform   all remaining countries equally weighted

Usage:
  python geo_interpolate.py \\
      --model models/ar_geo_twitter.pkl \\
      --nadi-dir datasets/nadi2020 \\
      [--k 5] [--nadi-split dev]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM, _ngrams


# ── Country centroids (lat, lon) ──────────────────────────────────────────────

CENTROIDS: dict[str, tuple[float, float]] = {
    # Arabic-speaking countries
    "AE": (24.0,  54.0),
    "BH": (26.0,  50.6),
    "DJ": (11.8,  42.6),
    "DZ": (28.0,   1.7),
    "EG": (26.8,  30.8),
    "IQ": (33.0,  44.0),
    "JO": (31.0,  36.0),
    "KM": (-11.6, 43.3),
    "KW": (29.3,  47.7),
    "LB": (33.9,  35.9),
    "LY": (26.3,  17.2),
    "MA": (31.8,  -7.1),
    "MR": (20.3, -10.9),
    "OM": (22.0,  57.0),
    "PS": (31.9,  35.2),
    "QA": (25.4,  51.2),
    "SA": (23.9,  45.1),
    "SD": (15.6,  32.5),
    "SO": ( 5.2,  46.2),
    "SY": (34.8,  38.9),
    "TN": (33.9,   9.5),
    "YE": (15.6,  48.5),
    # Portuguese-speaking countries
    "PT": (39.6,  -8.0),
    "BR": (-14.2, -51.9),
    "AO": (-11.2,  17.9),
    "MZ": (-18.7,  35.5),
    "CV": (16.0,  -24.0),
    # Spanish-speaking countries
    "ES": (40.5,  -3.7),
    "AR": (-38.4, -63.6),
    "MX": (23.6,  -102.6),
    "CO": (4.6,   -74.3),
    "PE": (-9.2,  -75.0),
    "VE": (6.4,   -66.6),
    "CL": (-35.7, -71.5),
    "EC": (-1.8,  -78.2),
    "BO": (-16.3, -63.6),
    "PY": (-23.4, -58.4),
    "UY": (-32.5, -55.8),
    "GT": (15.8,  -90.2),
    "HN": (15.2,  -86.2),
    "SV": (13.8,  -88.9),
    "NI": (12.9,  -85.2),
    "CR": (9.7,   -83.8),
    "PA": (8.5,   -80.8),
    "CU": (21.5,  -79.5),
    "DO": (19.0,  -70.7),
    # English-speaking countries
    "US": (37.1,  -95.7),
    "GB": (55.4,   -3.4),
    "AU": (-25.3,  133.8),
    "CA": (56.1,  -106.3),
    "NZ": (-40.9,  174.9),
    "IE": (53.1,   -8.2),
    "ZA": (-30.6,  22.9),
    "IN": (20.6,   78.9),
    "NG": ( 9.1,    8.7),
    "GH": ( 7.9,   -1.0),
    "KE": ( 0.0,   37.9),
    # German-speaking countries
    "DE": (51.2,   10.5),
    "AT": (47.5,   14.6),
    "CH": (46.8,    8.2),
    "LI": (47.1,    9.5),
    "LU": (49.8,    6.1),
    # French-speaking countries (European + African)
    "FR": (46.2,    2.2),
    "BE": (50.8,    4.5),
    "SN": (14.5,  -14.5),
    "CI": ( 7.5,   -5.5),
    "CM": ( 7.4,   12.4),
    "CD": (-4.0,   21.8),
    "MG": (-20.3,  44.9),
    # Russian-speaking countries (post-Soviet)
    "RU": (61.5,  105.3),
    "UA": (49.0,   31.5),
    "BY": (53.7,   27.9),
    "KZ": (48.0,   68.0),
    "KG": (41.2,   74.8),
    "MD": (47.4,   28.4),
    "AM": (40.1,   45.0),
    "GE": (42.3,   43.4),
    "AZ": (40.1,   47.6),
}


def haversine(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def geographic_weights(
    held_out: str,
    known: list[str],
    k: int | None = None,
    sigma_km: float | None = None,
) -> dict[str, float]:
    """Weights from held_out to each known country.

    k=None uses all known countries; otherwise restricts to the k nearest.
    sigma_km=None uses inverse-distance (1/d); otherwise uses Gaussian kernel
    exp(-d^2 / (2*sigma^2)).
    Returns a dict of normalized weights summing to 1.
    """
    if held_out not in CENTROIDS:
        raise ValueError(f"No centroid for {held_out!r}. Add it to CENTROIDS.")

    dists: list[tuple[float, str]] = []
    for c in known:
        if c == held_out or c not in CENTROIDS:
            continue
        dists.append((haversine(CENTROIDS[held_out], CENTROIDS[c]), c))

    dists.sort()
    if k is not None:
        dists = dists[:k]

    if sigma_km is not None:
        raw = {c: math.exp(-(d ** 2) / (2 * sigma_km ** 2)) for d, c in dists}
    else:
        raw = {c: 1.0 / (d + 1e-6) for d, c in dists}

    total = sum(raw.values())
    if total == 0:
        total = 1.0
    return {c: w / total for c, w in raw.items()}


# ── Mixture LM ────────────────────────────────────────────────────────────────

class MixtureNGramLM:
    """Linear probability mixture of NGramLMs.

    P(gram) = Σ w_i * P_i(gram)

    Duck-types NGramLM: Hierarchy.score_cells only needs .logprob(text).
    """

    def __init__(self, lms: list[NGramLM], weights: list[float]) -> None:
        total = sum(weights)
        self.lms = lms
        self.weights = [w / total for w in weights]
        self.order = lms[0].order

    def logprob(self, text: str) -> float:
        lp = 0.0
        for gram in _ngrams(text, self.order):
            # Arithmetic mixture of per-gram probabilities: P(g) = Σ w_i P_i(g).
            # _prob_with_backoff already returns a probability, so it must be
            # summed directly (no exp): log(Σ w_i P_i(g)) is then on the same
            # scale as a single cell's logprob.
            p = sum(
                w * lm._prob_with_backoff(gram, lm.order)
                for lm, w in zip(self.lms, self.weights)
            )
            lp += math.log(max(p, 1e-300))
        return lp


# ── Hierarchy builder ─────────────────────────────────────────────────────────

def build_interpolated_hierarchy(
    held_out: str,
    cells: list[Cell],
    k: int | None = 5,
    scheme: str = "geo",
    sigma_km: float | None = None,
    language: str = "ar",
) -> Hierarchy:
    """Build a hierarchy replacing held_out with a synthesized mixture cell.

    scheme: "geo" (inverse-distance, top-k), "nearest" (k=1), "uniform" (all equal),
            "gaussian" (Gaussian kernel, requires sigma_km)
    """
    known_cells = {c.region: c for c in cells if c.region != held_out}
    known = list(known_cells)

    if scheme == "geo":
        w_dict = geographic_weights(held_out, known, k=k)
    elif scheme == "nearest":
        w_dict = geographic_weights(held_out, known, k=1)
    elif scheme == "uniform":
        w_dict = {c: 1.0 / len(known) for c in known}
    elif scheme == "gaussian":
        if sigma_km is None:
            raise ValueError("sigma_km required for gaussian scheme")
        w_dict = geographic_weights(held_out, known, k=k, sigma_km=sigma_km)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    source_cells = [known_cells[c] for c in w_dict]
    weights_list = [w_dict[c] for c in w_dict]

    mixture_lm = MixtureNGramLM(
        lms=[c.lm for c in source_cells],
        weights=weights_list,
    )
    avg_prior = sum(c.log_prior for c in source_cells) / len(source_cells)
    interp_cell = Cell(
        language=language,
        region=held_out,
        lm=mixture_lm,
        log_prior=avg_prior,
    )

    hier = Hierarchy()
    hier.log_language_prior = {language: 0.0}
    for cell in cells:
        if cell.region != held_out:
            hier.add_cell(cell)
    hier.add_cell(interp_cell)
    return hier


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _predict(hier: Hierarchy, texts: list[str]) -> list[str | None]:
    return [hier.predict_dialect(t)[0][1] for t in texts]


def _f1(gold: list[str], pred: list[str | None], country: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == country and p == country)
    fp = sum(1 for g, p in zip(gold, pred) if g != country and p == country)
    fn = sum(1 for g, p in zip(gold, pred) if g == country and p != country)
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-12)


# ── Leave-one-out ─────────────────────────────────────────────────────────────

def _cache_logprobs(cells: list[Cell], texts: list[str]) -> dict[str, list[float]]:
    """Pre-compute logprob(text) for every real (non-mixture) cell.

    Returns {region: [logprob_per_text]}. Used to avoid redundant computation
    in the LOO loop — only the one held-out cell changes each iteration.
    """
    cache: dict[str, list[float]] = {}
    for cell in cells:
        if cell.region is None:
            continue
        cache[cell.region] = [cell.lm.logprob(t) for t in texts]
    return cache


def loo_evaluate(
    model_path: Path,
    nadi_dir: Path,
    nadi_split: str = "dev",
    k: int = 5,
) -> dict[str, dict]:
    """Run LOO evaluation for all countries present in both model and NADI split.

    Returns country -> {oracle_f1, geo_f1, nn_f1, dist_nn_km, recovery}.
    The special key "MACRO" contains unweighted averages.
    """
    from save_load import load as load_bundle
    from data.nadi import load_split

    bundle = load_bundle(model_path)
    cells: list[Cell] = bundle.hierarchy.cells
    print(f"Loaded: {len(cells)} cells from {model_path.name}", flush=True)

    examples = [e for e in load_split(nadi_dir, nadi_split) if e.country]
    texts = [e.text for e in examples]
    gold  = [e.country for e in examples]

    model_countries = {c.region for c in cells if c.region}
    nadi_countries  = set(gold)
    eval_countries  = sorted(model_countries & nadi_countries)

    print(f"Evaluating {len(eval_countries)} shared countries  |  {len(texts)} examples", flush=True)
    if model_countries - nadi_countries:
        print(f"  model-only (skipped): {sorted(model_countries - nadi_countries)}")
    if nadi_countries - model_countries:
        print(f"  NADI-only (no model): {sorted(nadi_countries - model_countries)}")
    print(flush=True)

    # Pre-compute all real-cell logprobs once — reused across LOO iterations
    print("Pre-computing cell logprobs ...", end=" ", flush=True)
    lp_cache = _cache_logprobs(cells, texts)
    print("done", flush=True)

    # Cell metadata for quick lookup
    cell_by_region = {c.region: c for c in cells if c.region}
    lang_prior = bundle.hierarchy.log_language_prior.get("ar", 0.0)

    def _score_cached(region: str, i: int) -> float:
        cell = cell_by_region[region]
        return lp_cache[region][i] + cell.log_prior + lang_prior

    def _score_mixture(mixture_lm: MixtureNGramLM, avg_prior: float, i: int) -> float:
        return mixture_lm.logprob(texts[i]) + avg_prior + lang_prior

    def predict_with_mixture(
        held_out: str,
        mixture_lm: MixtureNGramLM,
        avg_prior: float,
    ) -> list[str | None]:
        known_regions = [r for r in lp_cache if r != held_out]
        preds = []
        for i in range(len(texts)):
            scores: dict[str, float] = {r: _score_cached(r, i) for r in known_regions}
            scores[held_out] = _score_mixture(mixture_lm, avg_prior, i)
            preds.append(max(scores, key=scores.__getitem__))
        return preds

    # Oracle predictions from cached scores
    def predict_oracle() -> list[str | None]:
        preds = []
        for i in range(len(texts)):
            scores = {r: _score_cached(r, i) for r in lp_cache}
            preds.append(max(scores, key=scores.__getitem__))
        return preds

    print("Running oracle predictions ...", end=" ", flush=True)
    pred_oracle = predict_oracle()
    print("done", flush=True)
    print(flush=True)

    results: dict[str, dict] = {}

    for held_out in eval_countries:
        f1_oracle = _f1(gold, pred_oracle, held_out)

        known = [r for r in lp_cache if r != held_out]

        # Geo interpolation (k nearest neighbors)
        w_geo  = geographic_weights(held_out, known, k=k)
        lms_geo = [cell_by_region[c].lm for c in w_geo]
        wts_geo = list(w_geo.values())
        mix_geo = MixtureNGramLM(lms_geo, wts_geo)
        avg_prior_geo = sum(cell_by_region[c].log_prior for c in w_geo) / len(w_geo)
        f1_geo = _f1(gold, predict_with_mixture(held_out, mix_geo, avg_prior_geo), held_out)

        # Nearest-neighbor only (k=1)
        w_nn  = geographic_weights(held_out, known, k=1)
        lms_nn = [cell_by_region[c].lm for c in w_nn]
        mix_nn = MixtureNGramLM(lms_nn, list(w_nn.values()))
        avg_prior_nn = sum(cell_by_region[c].log_prior for c in w_nn) / len(w_nn)
        f1_nn = _f1(gold, predict_with_mixture(held_out, mix_nn, avg_prior_nn), held_out)

        # Distance to nearest model neighbor
        if held_out in CENTROIDS:
            dists = [(haversine(CENTROIDS[held_out], CENTROIDS[c]), c)
                     for c in known if c in CENTROIDS]
            dists.sort()
            dist_nn = dists[0][0] if dists else float("nan")
            nn_name = dists[0][1] if dists else "?"
        else:
            dist_nn, nn_name = float("nan"), "?"

        results[held_out] = {
            "oracle_f1":  f1_oracle,
            "geo_f1":     f1_geo,
            "nn_f1":      f1_nn,
            "dist_nn_km": dist_nn,
            "nn_country": nn_name,
            "recovery":   f1_geo / max(f1_oracle, 1e-6),
        }
        print(
            f"  {held_out:<3}  oracle={f1_oracle:.3f}  geo={f1_geo:.3f}  "
            f"nn={f1_nn:.3f}  dist={dist_nn:>6.0f}km  nn={nn_name}",
            flush=True,
        )

    # Macro averages. Exclude countries whose oracle F1 is ~0 (e.g. Bahrain,
    # which has no usable signal): a near-zero oracle makes per-country recovery
    # explode (geo_f1 / ~0), which would otherwise dominate the macro.
    # Recovery is reported as the ratio of macro F1s (geo / oracle), matching
    # the definition used in the paper, not the mean of per-country ratios.
    valid = [r for r in results.values() if r["oracle_f1"] > 0.01]
    n = len(valid)
    macro_oracle = sum(r["oracle_f1"] for r in valid) / n
    macro_geo    = sum(r["geo_f1"]    for r in valid) / n
    macro_nn     = sum(r["nn_f1"]     for r in valid) / n
    results["MACRO"] = {
        "oracle_f1":  macro_oracle,
        "geo_f1":     macro_geo,
        "nn_f1":      macro_nn,
        "recovery":   macro_geo / max(macro_oracle, 1e-9),
        "dist_nn_km": float("nan"),
        "nn_country": "—",
    }

    return results


def _print_results(results: dict[str, dict]) -> None:
    hdr = (f"{'':6}  {'oracle':>7}  {'geo_k5':>7}  "
           f"{'nearest':>7}  {'dist_km':>8}  {'recovery':>9}  nn")
    print(hdr)
    print("-" * len(hdr))
    for country, r in results.items():
        dist = f"{r['dist_nn_km']:>8.0f}" if not math.isnan(r["dist_nn_km"]) else f"{'—':>8}"
        rec  = f"{r['recovery']:>9.1%}" if isinstance(r["recovery"], float) else f"{'—':>9}"
        print(
            f"{country:<6}  "
            f"{r['oracle_f1']:>7.3f}  "
            f"{r['geo_f1']:>7.3f}  "
            f"{r['nn_f1']:>7.3f}  "
            f"{dist}  "
            f"{rec}  "
            f"{r['nn_country']}"
        )


# ── Kernel ablation ───────────────────────────────────────────────────────────

def kernel_ablation(
    model_path: Path,
    nadi_dir: Path,
    nadi_split: str = "dev",
    sigmas_km: list[float] | None = None,
) -> dict[str, float]:
    """Run LOO with Gaussian kernel for each sigma_km; return {sigma: macro_f1}.

    Also includes inverse-distance (sigma=None) and nearest-1 as references.
    """
    if sigmas_km is None:
        sigmas_km = [50, 100, 200, 500, 1000, 2000, 5000]

    from save_load import load as load_bundle
    from data.nadi import load_split

    bundle = load_bundle(model_path)
    cells = bundle.hierarchy.cells

    examples = [e for e in load_split(nadi_dir, nadi_split) if e.country]
    texts = [e.text for e in examples]
    gold  = [e.country for e in examples]

    model_countries = {c.region for c in cells if c.region}
    eval_countries  = sorted(model_countries & set(gold))

    print("Pre-computing logprobs ...", end=" ", flush=True)
    lp_cache = _cache_logprobs(cells, texts)
    print("done", flush=True)

    cell_by_region = {c.region: c for c in cells if c.region}
    lang_prior = bundle.hierarchy.log_language_prior.get("ar", 0.0)

    def _score_c(r, i):
        c = cell_by_region[r]
        return lp_cache[r][i] + c.log_prior + lang_prior

    def _score_mix(mix_lm, avg_prior, i):
        return mix_lm.logprob(texts[i]) + avg_prior + lang_prior

    def predict_mixture(held_out, mix_lm, avg_prior):
        known = [r for r in lp_cache if r != held_out]
        preds = []
        for i in range(len(texts)):
            scores = {r: _score_c(r, i) for r in known}
            scores[held_out] = _score_mix(mix_lm, avg_prior, i)
            preds.append(max(scores, key=scores.__getitem__))
        return preds

    def macro_f1_for_scheme(w_fn):
        f1s = []
        for held_out in eval_countries:
            if held_out == "BH":
                continue
            known = [r for r in lp_cache if r != held_out]
            w = w_fn(held_out, known)
            src = [cell_by_region[c] for c in w]
            mix = MixtureNGramLM([c.lm for c in src], list(w.values()))
            avg_p = sum(c.log_prior for c in src) / len(src)
            pred = predict_mixture(held_out, mix, avg_p)
            f1s.append(_f1(gold, pred, held_out))
        return sum(f1s) / len(f1s)

    results: dict[str, float] = {}

    print("inverse-distance ...", end=" ", flush=True)
    results["inv_dist"] = macro_f1_for_scheme(
        lambda h, k: geographic_weights(h, k, k=5))
    print(f"{results['inv_dist']:.3f}", flush=True)

    print("nearest-1 ...", end=" ", flush=True)
    results["nearest_1"] = macro_f1_for_scheme(
        lambda h, k: geographic_weights(h, k, k=1))
    print(f"{results['nearest_1']:.3f}", flush=True)

    for sigma in sigmas_km:
        print(f"gaussian sigma={sigma}km ...", end=" ", flush=True)
        results[f"gauss_{sigma}"] = macro_f1_for_scheme(
            lambda h, k, s=sigma: geographic_weights(h, k, sigma_km=s))
        print(f"{results[f'gauss_{sigma}']:.3f}", flush=True)

    return results


# ── True zero-shot (unseen countries) ─────────────────────────────────────────

def zero_shot_unseen(
    model_path: Path,
    nadi_dir: Path,
    unseen: list[str],
    nadi_split: str = "dev",
    k: int = 5,
) -> dict[str, dict]:
    """Evaluate on NADI countries absent from the trained model.

    For each unseen country, synthesizes a cell via geographic interpolation
    from all known model cells, then reports F1 for that country.
    Also reports fastText / GlotLID F1 for comparison (if available).
    """
    from save_load import load as load_bundle
    from data.nadi import load_split

    bundle = load_bundle(model_path)
    cells  = bundle.hierarchy.cells
    language = cells[0].language if cells else "ar"

    examples = [e for e in load_split(nadi_dir, nadi_split) if e.country]
    texts = [e.text for e in examples]
    gold  = [e.country for e in examples]

    known_regions = [c.region for c in cells if c.region]

    results: dict[str, dict] = {}
    for target in unseen:
        if target not in CENTROIDS:
            print(f"  {target}: no centroid, skipping")
            continue
        nadi_count = sum(1 for g in gold if g == target)
        if nadi_count == 0:
            print(f"  {target}: not in NADI {nadi_split}, skipping")
            continue

        w = geographic_weights(target, known_regions, k=k)
        src = [c for c in cells if c.region in w]
        mix = MixtureNGramLM([c.lm for c in src], [w[c.region] for c in src])
        avg_prior = sum(c.log_prior for c in src) / len(src)
        lang_prior = bundle.hierarchy.log_language_prior.get(language, 0.0)

        # Build extended hierarchy: all known cells + synthesized target
        ext_hier = Hierarchy()
        ext_hier.log_language_prior = {language: 0.0}
        for c in cells:
            ext_hier.add_cell(c)
        from hierarchy import Cell as _Cell
        ext_hier.add_cell(_Cell(language=language, region=target,
                                lm=mix, log_prior=avg_prior + lang_prior))

        pred = [ext_hier.predict_dialect(t)[0][1] for t in texts]
        f1 = _f1(gold, pred, target)

        nn_country = min(
            (c for c in known_regions if c in CENTROIDS),
            key=lambda c: haversine(CENTROIDS[target], CENTROIDS[c]),
        )
        dist_nn = haversine(CENTROIDS[target], CENTROIDS[nn_country])

        results[target] = {
            "geo_f1":     f1,
            "nadi_count": nadi_count,
            "dist_nn_km": dist_nn,
            "nn_country": nn_country,
        }
        print(f"  {target}: geo_f1={f1:.3f}  ({nadi_count} examples in NADI, "
              f"nn={nn_country} at {dist_nn:.0f}km)", flush=True)

    return results


# ── Multilingual LOO (DSL-TL) ─────────────────────────────────────────────────

def _load_dsltl(path: Path, lang: str) -> tuple[list[str], list[str]]:
    """Load a DSL-TL TSV and return (texts, labels) with normalised country codes."""
    label_map = {
        # Portuguese
        "PT": "PT", "PT-BR": "BR",
        # Spanish
        "ES": "ES", "ES-ES": "ES", "ES-AR": "AR",
        # English
        "EN": "EN", "EN-GB": "GB", "EN-US": "US",
    }
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw_label = parts[2].strip()
            if raw_label not in label_map:
                continue
            texts.append(parts[1].strip())
            labels.append(label_map[raw_label])
    return texts, labels


def multilingual_loo(
    pt_model: Path,
    es_model: Path,
    dsltl_dir: Path,
    k: int = 5,
) -> dict[str, dict]:
    """LOO evaluation for Portuguese and Spanish on DSL-TL dev."""
    from save_load import load as load_bundle

    results: dict[str, dict] = {}

    for lang_code, model_path, tsv_name, language in [
        ("pt", pt_model, "PT-DSL-TL/PT_dev.tsv", "pt"),
        ("es", es_model, "ES-DSL-TL/ES_dev.tsv", "es"),
    ]:
        tsv_path = dsltl_dir / tsv_name
        if not tsv_path.exists():
            print(f"  {tsv_name} not found, skipping")
            continue

        texts, gold = _load_dsltl(tsv_path, lang_code)
        eval_countries = sorted(set(gold))

        bundle = load_bundle(model_path)
        cells  = bundle.hierarchy.cells
        model_countries = {c.region for c in cells if c.region}

        overlap = sorted(model_countries & set(eval_countries))
        if not overlap:
            print(f"  {lang_code}: no overlap between model and DSL-TL labels")
            continue

        print(f"\n{lang_code.upper()} LOO on DSL-TL dev: {len(texts)} examples, "
              f"labels={eval_countries}, model covers {sorted(model_countries)[:5]}...")

        print("  Pre-computing logprobs ...", end=" ", flush=True)
        lp_cache = _cache_logprobs(cells, texts)
        print("done", flush=True)

        cell_by_region = {c.region: c for c in cells if c.region}
        lang_prior = bundle.hierarchy.log_language_prior.get(language, 0.0)

        def _sc(r, i):
            c = cell_by_region[r]
            return lp_cache[r][i] + c.log_prior + lang_prior

        def _sm(mix_lm, avg_prior, i):
            return mix_lm.logprob(texts[i]) + avg_prior + lang_prior

        def predict_oracle_lang():
            preds = []
            for i in range(len(texts)):
                scores = {r: _sc(r, i) for r in lp_cache}
                preds.append(max(scores, key=scores.__getitem__))
            return preds

        def predict_mix_lang(held_out, mix_lm, avg_prior):
            known_r = [r for r in lp_cache if r != held_out]
            preds = []
            for i in range(len(texts)):
                scores = {r: _sc(r, i) for r in known_r}
                scores[held_out] = _sm(mix_lm, avg_prior, i)
                preds.append(max(scores, key=scores.__getitem__))
            return preds

        pred_oracle = predict_oracle_lang()

        for held_out in overlap:
            f1_oracle = _f1(gold, pred_oracle, held_out)
            known = [r for r in lp_cache if r != held_out]
            if not known:
                continue

            w_geo = geographic_weights(held_out, known, k=min(k, len(known)))
            src   = [cell_by_region[c] for c in w_geo]
            mix   = MixtureNGramLM([c.lm for c in src], list(w_geo.values()))
            avg_p = sum(c.log_prior for c in src) / len(src)
            f1_geo = _f1(gold, predict_mix_lang(held_out, mix, avg_p), held_out)

            nn_c = min((c for c in known if c in CENTROIDS),
                       key=lambda c: haversine(CENTROIDS[held_out], CENTROIDS[c]),
                       default="?")
            dist_nn = haversine(CENTROIDS[held_out], CENTROIDS[nn_c]) if nn_c != "?" else float("nan")

            key = f"{lang_code.upper()}-{held_out}"
            results[key] = {
                "oracle_f1":  f1_oracle,
                "geo_f1":     f1_geo,
                "dist_nn_km": dist_nn,
                "nn_country": nn_c,
                "recovery":   f1_geo / max(f1_oracle, 1e-6),
            }
            print(f"  {key:<10}  oracle={f1_oracle:.3f}  geo={f1_geo:.3f}  "
                  f"dist={dist_nn:.0f}km  nn={nn_c}", flush=True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",       type=Path, default=Path("models/ar_geo_twitter.pkl"))
    ap.add_argument("--nadi-dir",    type=Path)
    ap.add_argument("--nadi-split",  default="dev")
    ap.add_argument("--k",           type=int,  default=5)
    ap.add_argument("--kernel-ablation", action="store_true",
                    help="Run Gaussian kernel ablation across sigma values")
    ap.add_argument("--zero-shot",   nargs="+", metavar="CC",
                    help="True zero-shot for these country codes (e.g. MR SO)")
    ap.add_argument("--multilingual", action="store_true",
                    help="Run multilingual LOO on DSL-TL for PT and ES")
    ap.add_argument("--dsltl-dir",   type=Path,
                    default=Path("datasets/DSL-TL/DSL-TL-Corpus"))
    ap.add_argument("--pt-model",    type=Path, default=Path("models/pt_cctld.pkl"))
    ap.add_argument("--es-model",    type=Path, default=Path("models/es_cctld.pkl"))
    args = ap.parse_args(argv)

    if args.kernel_ablation:
        print("=== Kernel Ablation ===")
        results_k = kernel_ablation(args.model, args.nadi_dir, args.nadi_split)
        print("\nscheme              macro-F1")
        print("-" * 32)
        for k2, v in results_k.items():
            print(f"  {k2:<20}  {v:.3f}")
        return 0

    if args.zero_shot:
        print(f"=== True Zero-Shot: {args.zero_shot} ===")
        results_z = zero_shot_unseen(
            args.model, args.nadi_dir, args.zero_shot, args.nadi_split, k=args.k)
        return 0

    if args.multilingual:
        print("=== Multilingual LOO (DSL-TL) ===")
        results_m = multilingual_loo(
            args.pt_model, args.es_model, args.dsltl_dir, k=args.k)
        print("\n--- Summary ---")
        for key, r in results_m.items():
            print(f"  {key:<12} oracle={r['oracle_f1']:.3f}  "
                  f"geo={r['geo_f1']:.3f}  recovery={r['recovery']:.1%}")
        return 0

    if args.nadi_dir is None:
        ap.error("--nadi-dir required for default LOO mode")

    results = loo_evaluate(args.model, args.nadi_dir, args.nadi_split, k=args.k)

    print(f"\n{'='*72}")
    print("Leave-One-Out: Geographic Interpolation vs Oracle")
    print(f"  model: {args.model}  |  eval: NADI {args.nadi_split}  |  k={args.k}")
    print(f"{'='*72}")
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
