"""Linguistic-geographic correlation: cosine similarity of n-gram profiles vs.
Haversine distance, across all available language models.

Produces paper2/figures/corr_plot.pdf with one panel per language.

Usage:
  python -m analysis.correlation_plot [--langs ar es pt fr de en ru]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from save_load import load as load_bundle
from geo_interpolate import CENTROIDS, haversine
from ngram_lm import NGramLM
from analysis.cctld_loo import load_country_docs, LANG_DIRS


# Lightweight stand-in for hierarchy.Cell (only needs .region and .lm here).
class _ProfileCell:
    def __init__(self, region: str, lm: NGramLM) -> None:
        self.region = region
        self.lm = lm


# All families: (cc_code, label, color, evolutionary_type).
# cc_code=None means Arabic, loaded from the Twitter .pkl bundle; all others
# are built from the same raw CC-TLD data the LOO uses, so r and recovery are
# computed over identical country sets.
AR_PKL = "models/ar_geo_twitter.pkl"
ALL_MODELS = [
    (None, "Arabic",    "#2166ac", "in-situ"),
    ("de", "German",    "#053061", "in-situ"),
    ("ru", "Russian",   "#762a83", "mixed"),
    ("fr", "French",    "#5aae61", "mixed"),
    ("pt", "Portuguese","#4dac26", "colonial"),
    ("en", "English",   "#f4a582", "colonial"),
    ("es", "Spanish",   "#d6604d", "colonial"),
]


def _cells_for(cc_code: str | None, order: int = 4) -> list:
    """Return profile cells for one family, from the same data the LOO uses."""
    if cc_code is None:  # Arabic: from the Twitter bundle
        bundle = load_bundle(ROOT / AR_PKL)
        return [c for c in bundle.hierarchy.cells if c.region]
    docs = load_country_docs(LANG_DIRS[cc_code])
    return [_ProfileCell(country, NGramLM(order=order).fit(texts))
            for country, texts in sorted(docs.items())]


# ── N-gram profile vector ─────────────────────────────────────────────────────

def profile_vector(lm, vocab: list[str]) -> np.ndarray:
    counts = lm.ngram_counts[-1]
    v = np.array([counts.get(g, 0.0) for g in vocab], dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def build_vocab(cells) -> list[str]:
    keys: set[str] = set()
    for cell in cells:
        keys.update(cell.lm.ngram_counts[-1].keys())
    return sorted(keys)


def pairwise_cosine(cells) -> tuple[np.ndarray, np.ndarray]:
    vocab = build_vocab(cells)
    vecs = {c.region: profile_vector(c.lm, vocab) for c in cells if c.region}
    regions = [c.region for c in cells if c.region and c.region in CENTROIDS]

    dists, sims = [], []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1, r2 = regions[i], regions[j]
            d = haversine(CENTROIDS[r1], CENTROIDS[r2])
            s = float(np.dot(vecs[r1], vecs[r2]))
            dists.append(d)
            sims.append(s)
    return np.array(dists), np.array(sims)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_language(ax, cells, lang_label: str, color: str, n_countries: int) -> float:
    dists, sims = pairwise_cosine(cells)
    if len(dists) < 3:
        ax.set_title(f"{lang_label}\n(n={n_countries}, insufficient pairs)", fontsize=9)
        return float("nan")
    r, p = stats.pearsonr(dists, sims)

    ax.scatter(dists / 1000, sims, s=14, alpha=0.55, color=color, linewidths=0)
    m, b = np.polyfit(dists / 1000, sims, 1)
    x_line = np.linspace(0, max(dists) / 1000, 100)
    ax.plot(x_line, m * x_line + b, color=color, linewidth=1.4, alpha=0.85)

    ax.set_title(f"{lang_label} ({n_countries})", fontsize=9)
    ax.set_xlabel("Distance (1000 km)", fontsize=7)
    ax.set_ylabel("Cosine sim.", fontsize=7)
    ax.tick_params(labelsize=6)

    p_str = "$p{<}0.001$" if p < 0.001 else (f"$p{{=}}{p:.2f}$" if p > 0.05 else "$p{<}0.05$")
    ax.annotate(f"$r{{=}}{r:+.2f}$\n{p_str}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    return r


def main(lang_codes: list[str] | None = None) -> dict[str, float]:
    out_path = ROOT / "paper2" / "figures" / "corr_plot.pdf"

    # Filter to requested languages
    def _match(entry) -> bool:
        cc_code, label = entry[0], entry[1]
        if lang_codes is None:
            return True
        code = cc_code if cc_code is not None else "ar"
        return code in lang_codes
    models_to_plot = [m for m in ALL_MODELS if _match(m)]

    if not models_to_plot:
        print("No families selected.")
        return {}

    n = len(models_to_plot)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.4),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.42, hspace=0.55,
                        left=0.07, right=0.97, top=0.92, bottom=0.12)

    rs: dict[str, float] = {}
    for idx, (cc_code, label, color, evo_type) in enumerate(models_to_plot):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        cells = _cells_for(cc_code)
        n_countries = len(cells)
        r = plot_language(ax, cells, label, color, n_countries)
        rs[label] = r
        evo_marker = {"in-situ": "●", "mixed": "◆", "colonial": "▲"}[evo_type]
        print(f"{evo_marker} {label:12s} ({n_countries:2d} countries)  r = {r:+.3f}")

    # Hide empty axes and center any incomplete last row
    n_used = len(models_to_plot)
    n_empty = nrows * ncols - n_used
    for idx in range(n_used, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if n_empty > 0 and nrows > 1:
        last_row = nrows - 1
        n_last = ncols - n_empty
        # shift = half the empty slots, in axes-fraction units
        pos0 = axes[last_row][0].get_position()
        pos1 = axes[last_row][1].get_position()
        col_step = pos1.x0 - pos0.x0
        shift = (n_empty / 2.0) * col_step
        for col in range(n_last):
            ax = axes[last_row][col]
            p = ax.get_position()
            ax.set_position([p.x0 + shift, p.y0, p.width, p.height])

    fig.suptitle("Geographic distance vs. n-gram profile cosine similarity",
                 fontsize=10, y=0.98)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    return rs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=None,
                    help="Language codes to include (default: all available)")
    args = ap.parse_args()
    main(args.langs)
