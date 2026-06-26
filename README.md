# Geographic n-gram interpolation for zero-shot dialect ID

Code behind *"Can Geography Stand In for Labeled Data in Arabic Dialect
Identification?"* — see [`paper2/`](paper2/) for the full writeup.

**Idea:** for a dialect/variety with no labeled data, build its language
model as a distance-weighted mixture of its `k` nearest *known* varieties
(Tobler's first law of geography). We test this on Arabic dialect ID (NADI
2020) and stress-test it on six more language families (German, Russian,
English, Portuguese, Spanish, French) to check whether the
geography–similarity correlation can predict, ahead of time, when the trick
will work. It doesn't: recovery is a usable proxy everywhere (26–63% of
oracle macro F1 with zero target labels), but no single correlation
threshold lets you screen a new language family in advance.

## Layout

- `ngram_lm.py` — character n-gram LM with interpolated Kneser-Ney smoothing.
- `hierarchy.py` — (language, country) cell tree; log-prior bookkeeping and inference.
- `geo_interpolate.py` — the core method: builds a synthesized model for an unseen country from its `k` nearest known countries, weighted by inverse Haversine distance. Runs the leave-one-out oracle/geo/nearest/uniform comparison for Arabic against NADI.
- `train_geo.py` — trains a (language, country) hierarchy from geo-labeled text (Arap-Tweet, HF Arabic Twitter, CC-TLD mc4, NADI gold, or DSLCC).
- `eval_nadi.py` — main Arabic experiment: geo-interpolation vs. fastText/GlotLID baselines vs. oracle on NADI 2020.
- `eval_dialect.py` — same protocol generalized to other languages (Spanish, Portuguese, ... via DSLCC/DSL-TL).
- `eval_lince.py` — code-switching evaluation on LinCE (token-level, region-conditioned Viterbi decoding).
- `viterbi.py` — token-level decoder with region-conditioned transition costs, for code-switched text.
- `bootstrap.py` — self-training expansion for low-resource cells.
- `analysis/` — the six-family stress test: `cctld_loo.py` (per-language leave-one-out sweep), `correlation_plot.py` (geography vs. similarity correlation, `paper2/figures/corr_plot.pdf`), `annotation_curve.py` (labels-needed-to-beat-zero-shot curve).
- `baselines/` — fastText / GlotLID zero-shot baselines.
- `data/` — dataset loaders (NADI, MADAR, DSL-TL, Arap-Tweet, CC-TLD mc4, LinCE), with download instructions in each module's docstring. Datasets themselves are not committed; these loaders fetch/read them locally into `datasets/`.

`models/` and `datasets/` are git-ignored (several GB total, and NADI/MADAR
are gated behind registration forms — see `data/nadi.py` and `data/madar.py`
docstrings for access instructions). Regenerate them locally with
`train_geo.py` and the loaders in `data/`.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the Arabic result

```bash
# 1. Train per-country LMs from geo-labeled Arabic Twitter data
python train_geo.py --language ar --source aratweet_hf --stream \
    --out models/ar_geo_twitter.pkl

# 2. Get NADI 2020 (see data/nadi.py docstring for the access form),
#    unzip into datasets/nadi2020/

# 3. Run the leave-one-out geo-interpolation evaluation against NADI 2020,
#    including baselines (fastText, GlotLID) and the oracle upper bound
python eval_nadi.py --nadi-dir datasets/nadi2020 \
    --aratweet-model models/ar_geo_twitter.pkl
```

## Reproducing the six-family stress test

```bash
# Train a CC-TLD model for, e.g., German
python train_geo.py --language de --source cc_tld --stream --out models/de_cctld.pkl

# Per-language leave-one-out recovery
python -m analysis.cctld_loo --lang es

# Geography vs. n-gram similarity correlation across all trained languages
python -m analysis.correlation_plot --langs ar es pt fr de en ru

# Labeled-tweets-needed-to-beat-zero-shot curve (Arabic)
python -m analysis.annotation_curve
```

## Known limitations / research traps to remember

- **Country ≠ dialect.** Borders cut through dialect continua (Levantine, Maghrebi, Peninsular Arabic). Evaluate on speaker-held-out splits, not document splits.
- **Twitter geolocation is biased.** Geo-labeled tweets model "what people in country X tweet about" as much as dialect; diaspora users tweet in their heritage dialect from abroad.
- **CC-TLD is a noisy geographic label.** A `.de` domain doesn't guarantee native German content.
- **Geography is a coarse proxy, not a causal one.** Contact history can override raw distance (e.g., Lebanon vs. Palestine in the Arabic results) — see the paper's Discussion and Limitations sections for the full list of caveats.

See [`paper2/main.tex`](paper2/main.tex) for the full writeup, including the
cross-language correlation analysis and all per-country tables.
