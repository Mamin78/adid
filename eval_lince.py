"""Train + evaluate tinylid on LinCE LID Spa-Eng dev split.

Reports:
  - Token-level accuracy and macro-F1 over {es, en, other}
  - Switch-point F1: at boundaries where the gold label changes from one
    sentence-internal language to another, did we predict a switch at the
    same position?
  - Three model variants for ablation:
      flat          : one LM per language, argmax per token (no Viterbi)
      flat+viterbi  : one LM per language, Viterbi with a uniform switch penalty
      hier+viterbi  : full hierarchy (here: language only since LinCE has no
                       region labels) — included to keep the API consistent
                       and ready for dialect-labeled data later.

Run:
    python eval_lince.py --root ./datasets/lince
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from data.lince import TaggedSentence, documents_by_language, load_split
from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM
from viterbi import TransitionModel, viterbi_decode


LABELS = ["es", "en", "other"]


def train_other_lm(train: list[TaggedSentence]) -> NGramLM:
    """Train an 'other' LM on punctuation/numbers/NE tokens so the model has
    somewhere to put them. Without this, punct gets force-classified as es/en."""
    pieces: list[str] = []
    for s in train:
        run: list[str] = []
        for tok, lab in zip(s.tokens, s.labels):
            if lab == "other":
                run.append(tok)
            elif run:
                pieces.append(" ".join(run))
                run = []
        if run:
            pieces.append(" ".join(run))
    return NGramLM(order=3).fit(pieces or ["."])


def build_hierarchy(train: list[TaggedSentence], order: int = 4) -> Hierarchy:
    docs = documents_by_language(train)
    hier = Hierarchy()
    counts = {}
    for lang in ("es", "en"):
        lm = NGramLM(order=order).fit(docs[lang] or [" "])
        hier.add_cell(Cell(language=lang, region=None, lm=lm))
        counts[(lang, None)] = max(len(docs[lang]), 1)
    other_lm = train_other_lm(train)
    hier.add_cell(Cell(language="other", region=None, lm=other_lm))
    counts[("other", None)] = max(sum(s.labels.count("other") for s in train), 1)
    hier.fit_priors_from_counts(counts)
    return hier


def predict_flat(hier: Hierarchy, sent: TaggedSentence) -> list[str]:
    out = []
    for tok in sent.tokens:
        best, best_lp = None, float("-inf")
        for cell in hier.cells:
            lp = cell.lm.logprob(tok) + cell.log_prior + hier.log_language_prior.get(cell.language, 0.0)
            if lp > best_lp:
                best_lp, best = lp, cell.language
        out.append(best)
    return out


def predict_viterbi(hier: Hierarchy, sent: TaggedSentence, trans: TransitionModel) -> list[str]:
    path = viterbi_decode(sent.tokens, hier, trans)
    return [c.language for c in path]


def score(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    n_correct = 0
    n_total = 0
    sw_tp = sw_fp = sw_fn = 0

    for g_sent, p_sent in zip(gold, pred):
        for g, p in zip(g_sent, p_sent):
            n_total += 1
            if g == p:
                n_correct += 1
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1
        for i in range(1, len(g_sent)):
            g_sw = g_sent[i] != g_sent[i - 1] and "other" not in (g_sent[i], g_sent[i - 1])
            p_sw = p_sent[i] != p_sent[i - 1] and "other" not in (p_sent[i], p_sent[i - 1])
            if g_sw and p_sw:
                sw_tp += 1
            elif p_sw and not g_sw:
                sw_fp += 1
            elif g_sw and not p_sw:
                sw_fn += 1

    f1s = {}
    for lab in LABELS:
        p = tp[lab] / max(tp[lab] + fp[lab], 1)
        r = tp[lab] / max(tp[lab] + fn[lab], 1)
        f1s[lab] = 2 * p * r / max(p + r, 1e-12)
    macro_f1 = sum(f1s.values()) / len(LABELS)

    sw_p = sw_tp / max(sw_tp + sw_fp, 1)
    sw_r = sw_tp / max(sw_tp + sw_fn, 1)
    sw_f1 = 2 * sw_p * sw_r / max(sw_p + sw_r, 1e-12)

    return {
        "accuracy": n_correct / max(n_total, 1),
        "f1_per_label": f1s,
        "macro_f1": macro_f1,
        "switch_f1": sw_f1,
        "n_tokens": n_total,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="LinCE root with lid_spaeng/{train,dev}.conll")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)

    print("Loading splits...")
    train = load_split(args.root, "train")
    dev = load_split(args.root, "dev")
    if args.max_train:
        train = train[: args.max_train]
    print(f"  train sents: {len(train)}, dev sents: {len(dev)}")

    print(f"Building hierarchy (order={args.order})...")
    hier = build_hierarchy(train, order=args.order)

    gold = [s.labels for s in dev]

    print("Evaluating: flat argmax")
    pred_flat = [predict_flat(hier, s) for s in dev]
    r1 = score(gold, pred_flat)

    print("Evaluating: flat + Viterbi (uniform switch penalty)")
    trans_uniform = TransitionModel(
        stay=-0.05, same_language_switch=-3.0,
        coherent_pair_switch=-3.0, incoherent_switch=-3.0,
    )
    pred_vit = [predict_viterbi(hier, s, trans_uniform) for s in dev]
    r2 = score(gold, pred_vit)

    print("Evaluating: hier + Viterbi (region-aware, no-op without dialect labels)")
    trans_hier = TransitionModel()
    pred_hier = [predict_viterbi(hier, s, trans_hier) for s in dev]
    r3 = score(gold, pred_hier)

    print()
    print(f"{'model':<24} {'acc':>7} {'macroF1':>9} {'switchF1':>10}  per-label F1")
    for name, r in [
        ("flat", r1),
        ("flat+viterbi", r2),
        ("hier+viterbi", r3),
    ]:
        per = " ".join(f"{l}={r['f1_per_label'][l]:.3f}" for l in LABELS)
        print(f"{name:<24} {r['accuracy']:>7.3f} {r['macro_f1']:>9.3f} {r['switch_f1']:>10.3f}  {per}")
    print(f"\nTokens evaluated: {r1['n_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
