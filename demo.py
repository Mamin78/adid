"""End-to-end demo on synthetic data.

Purpose: prove the wiring works and let you eyeball the behavior. The "data"
here is intentionally caricatured (not real dialect samples) so the demo runs
offline. Replace `_synthetic_corpus()` with MADAR/NADI/LinCE loaders to do
actual research.
"""

from __future__ import annotations

from bootstrap import BootstrapConfig, bootstrap_cell
from hierarchy import Cell, Hierarchy
from ngram_lm import NGramLM
from viterbi import TransitionModel, viterbi_decode


def _synthetic_corpus() -> dict[tuple[str, str | None], list[str]]:
    """Toy data. Each cell has a distinctive lexical signature."""
    return {
        ("en", "US"): [
            "y'all gonna love this new spot downtown",
            "i'm fixing to head out, catch you later",
            "that's awesome dude, no problem at all",
        ] * 10,
        ("en", "GB"): [
            "brilliant mate, fancy a cuppa later",
            "the queue at the chemist was absolutely mental",
            "i reckon we should take the lift, innit",
        ] * 10,
        ("es", "MX"): [
            "ahorita vengo wey, no manches",
            "qué padre está ese carro, está chido",
            "vamos por unos tacos al pastor compa",
        ] * 10,
        ("es", "AR"): [
            "che boludo, vamos a tomar unos mates",
            "está re copado el laburo nuevo viste",
            "no seas pelotudo, dale que llegamos tarde",
        ] * 10,
    }


def build_demo_hierarchy() -> tuple[Hierarchy, TransitionModel]:
    corpus = _synthetic_corpus()
    hier = Hierarchy()
    counts: dict[tuple[str, str | None], int] = {}
    for (lang, region), texts in corpus.items():
        lm = NGramLM(order=4).fit(texts)
        hier.add_cell(Cell(language=lang, region=region, lm=lm))
        counts[(lang, region)] = len(texts)
    hier.fit_priors_from_counts(counts)

    trans = TransitionModel()
    # Geographic coherence: Mexican Spanish + US English co-occur (border, diaspora);
    # Argentine Spanish + British English do not.
    trans.add_coherent_pair(("es", "MX"), ("en", "US"))
    return hier, trans


def demo_language_and_dialect_id(hier: Hierarchy) -> None:
    print("=== Document-level LID + dialect ID ===\n")
    samples = [
        "y'all want some tacos al pastor or what",  # code-switch-y
        "che boludo qué onda",                      # AR Spanish
        "brilliant mate, well done",                # GB English
        "ahorita vengo wey",                        # MX Spanish
    ]
    for s in samples:
        lang, lang_post = hier.predict_language(s)
        cell, dial_post = hier.predict_dialect(s)
        top3 = sorted(dial_post.items(), key=lambda kv: -kv[1])[:3]
        print(f"  text: {s!r}")
        print(f"    lang: {lang}  (post: {{ {', '.join(f'{k}: {v:.2f}' for k,v in lang_post.items())} }})")
        print(f"    dialect top-3: {top3}\n")


def demo_token_level(hier: Hierarchy, trans: TransitionModel) -> None:
    print("=== Token-level decoding with region-aware transitions ===\n")
    tokens = "yo wey i'm fixing to go por unos tacos".split()
    path = viterbi_decode(tokens, hier, trans)
    for tok, cell in zip(tokens, path):
        print(f"  {tok:>12s} -> {cell.language}-{cell.region}")
    print()


def demo_bootstrap(hier: Hierarchy) -> None:
    print("=== Low-resource bootstrap (synthetic) ===\n")
    # Pretend we only have 3 seed examples for a new cell (es, CO).
    seed = [
        "parcero qué más, todo bien o qué",
        "esa vaina está muy chévere, hermano",
        "vamos a comerte una bandeja paisa",
    ]
    # Unlabeled pool: mix of everything else plus a few more CO-ish lines.
    pool = [
        "parcero vamos por una arepa",
        "qué chévere parce, bacano todo",
        "brilliant mate, well done",
        "che boludo qué onda",
        "ahorita vengo wey",
    ]
    cell, delta = bootstrap_cell(
        language="es",
        region="CO",
        seed_texts=seed * 3,  # tiny seed, repeated for held-out split
        unlabeled_pool=pool,
        hierarchy=hier,
        config=BootstrapConfig(order=3, confidence_threshold=0.5, max_pseudo_examples=10),
    )
    print(f"  Built cell ({cell.language}, {cell.region}); held-out logprob delta = {delta:+.3f}")
    print(f"  ({'expansion helped' if delta >= 0 else 'kept seed-only model'})\n")


if __name__ == "__main__":
    hier, trans = build_demo_hierarchy()
    demo_language_and_dialect_id(hier)
    demo_token_level(hier, trans)
    demo_bootstrap(hier)
