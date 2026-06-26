"""DSL Corpus Collection (DSLCC) and DSL-TL loaders.

Two closely-related datasets are supported:

--- DSLCC v3 / v4 (Discriminating Similar Languages Corpus Collection) ---

Journalistic text (20-100 tokens) labeled by country of origin.
Each variety has 18,000 training + 2,000 dev instances.

Format (TSV, no header):
    text <TAB> label

Label format: language-COUNTRY, e.g.:
    Spanish:    es-AR, es-ES, es-MX, es-PE, es-CL  (v4 has 5; v3 has 3)
    Portuguese: pt-BR, pt-PT
    French:     fr-BE, fr-CA, fr-FR
    German:     de-AT, de-CH, de-DE

Download (free, no registration):
    http://ttg.uni-saarland.de/resources/DSLCC/
    Pick v3.0 or v4.0 zip and unzip into e.g. datasets/dslcc/.

--- DSL-TL (True Labels, 2023) ---

Human-annotated variety identification with native-speaker labels.
Covers three languages with 2-3 varieties each:

    Portuguese: PT-BR (Brazilian), PT-PT (European), PT (ambiguous)
    Spanish:    ES-ES (Spain), ES-AR (Argentina), ES (ambiguous)
    English:    EN-US (American), EN-GB (British), EN (ambiguous)

Format (TSV, 3 columns, NO header):
    sentence_id <TAB> text <TAB> label

Ambiguous labels (PT, ES, EN with no country suffix) are filtered out
by default (skip_ambiguous=True).

Download (GitHub, free, already included):
    git clone https://github.com/LanguageTechnologyLab/DSL-TL
    Files are in DSL-TL-Corpus/PT-DSL-TL/, ES-DSL-TL/, EN-DSL-TL/

--- How to use in the paper ---

Arabic:     NADI (primary) + MADAR (cross-domain)
Spanish:    DSL-TL ES (ES-ES vs ES-AR) — 2 clear varieties
Portuguese: DSL-TL PT (PT-BR vs PT-PT) — 2 clear varieties
English:    DSL-TL EN (EN-US vs EN-GB) — 2 clear varieties (bonus)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DSLExample:
    text: str
    lang: str      # e.g. "es"
    country: str   # ISO-2, e.g. "AR"
    raw_label: str # original label, e.g. "es-AR" or "EP"


# ── DSLCC ─────────────────────────────────────────────────────────────────────

def _parse_dslcc_label(label: str) -> tuple[str, str]:
    """Parse 'es-AR' → ('es', 'AR'). Returns ('', label) if unrecognized."""
    label = label.strip()
    if "-" in label:
        parts = label.split("-", 1)
        return parts[0].lower(), parts[1].upper()
    return "", label.upper()


def _iter_dslcc_file(path: Path, filter_lang: str | None) -> Iterator[DSLExample]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            raw_label = parts[-1].strip()  # label is last column
            lang, country = _parse_dslcc_label(raw_label)
            if filter_lang and lang and lang != filter_lang.lower():
                continue
            if text and country:
                yield DSLExample(text=text, lang=lang, country=country, raw_label=raw_label)


def _find_dslcc_file(root: Path, split: str) -> Path | None:
    patterns = [
        f"task1-{split}.txt", f"task-{split}.txt",
        f"{split}.txt", f"*{split}*.txt",
    ]
    for pat in patterns:
        found = sorted(root.glob(pat))
        if found:
            return found[0]
    return None


def load_dslcc(
    root: str | Path,
    split: str = "train",
    language: str | None = None,
) -> list[DSLExample]:
    """Load a DSLCC split, optionally filtered to one language.

    Args:
        root:     Directory containing DSLCC .txt files (or a language subdirectory).
        split:    'train', 'dev', or 'test'.
        language: ISO 639-1 code ('es', 'pt', 'fr', 'de') to filter, or None for all.

    Returns:
        List of DSLExample.

    Example:
        es_train = load_dslcc("datasets/dslcc", "train", language="es")
    """
    root = Path(root)

    # Check for per-language subdirectory first
    if language:
        lang_dir = root / language
        if lang_dir.is_dir():
            root = lang_dir

    path = _find_dslcc_file(root, split)
    if path is None:
        raise FileNotFoundError(
            f"No DSLCC {split} file found under {root}. "
            "Download from http://ttg.uni-saarland.de/resources/DSLCC/"
        )
    return list(_iter_dslcc_file(path, language))


def documents_by_country_dslcc(examples: list[DSLExample]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in examples:
        if ex.country:
            out.setdefault(ex.country, []).append(ex.text)
    return out


# ── DSL-TL (True Labels, multi-language) ─────────────────────────────────────

# DSL-TL label → (lang_code, ISO-2 country)
# Labels with no country suffix (PT, ES, EN) are ambiguous — filtered by default.
_DSLTL_LABEL_MAP: dict[str, tuple[str, str]] = {
    "PT-BR": ("pt", "BR"), "PT-PT": ("pt", "PT"),
    "ES-ES": ("es", "ES"), "ES-AR": ("es", "AR"),
    "EN-US": ("en", "US"), "EN-GB": ("en", "GB"),
}

# DSL-TL uses 3 columns: sentence_id <TAB> text <TAB> label (no header row)
# Some files have quoted text fields.
_DSLTL_LANG_FILES = {
    "pt": ("PT-DSL-TL", "PT"),
    "es": ("ES-DSL-TL", "ES"),
    "en": ("EN-DSL-TL", "EN"),
}


def _iter_dsltl_file(path: Path, language: str, skip_ambiguous: bool) -> Iterator[DSLExample]:
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for row in reader:
            if len(row) < 3:
                continue
            # id, text, label  (no header in DSL-TL files)
            text = row[1].strip()
            raw_label = row[2].strip()
            if raw_label not in _DSLTL_LABEL_MAP:
                if skip_ambiguous:
                    continue
                # ambiguous: assign as the language-only (no country)
                lang_out = language.lower()
                country = ""
            else:
                lang_out, country = _DSLTL_LABEL_MAP[raw_label]
            if text and country:
                yield DSLExample(text=text, lang=lang_out, country=country, raw_label=raw_label)


def load_dsltl(
    root: str | Path,
    split: str = "train",
    language: str = "pt",
    skip_ambiguous: bool = True,
) -> list[DSLExample]:
    """Load a DSL-TL split for a given language.

    Args:
        root:           Root of the cloned DSL-TL repo (the directory that
                        contains DSL-TL-Corpus/).
        split:          'train' or 'dev'.
        language:       'pt', 'es', or 'en'.
        skip_ambiguous: If True, discard rows with no country suffix
                        (PT, ES, EN alone). Default True.

    Returns:
        List of DSLExample. Country codes:
          Portuguese → BR, PT
          Spanish    → ES (Spain), AR
          English    → US, GB

    Path convention (matches the cloned repo layout):
        <root>/DSL-TL-Corpus/PT-DSL-TL/PT_train.tsv
        <root>/DSL-TL-Corpus/ES-DSL-TL/ES_train.tsv
        <root>/DSL-TL-Corpus/EN-DSL-TL/EN_train.tsv
    """
    root = Path(root)
    lang = language.lower()
    if lang not in _DSLTL_LANG_FILES:
        raise ValueError(f"DSL-TL supports languages: {list(_DSLTL_LANG_FILES)}")

    subdir, prefix = _DSLTL_LANG_FILES[lang]
    # Try the standard nested layout first, then flat
    candidates = [
        root / "DSL-TL-Corpus" / subdir / f"{prefix}_{split}.tsv",
        root / subdir / f"{prefix}_{split}.tsv",
        root / f"{prefix}_{split}.tsv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No DSL-TL {lang} {split} file found under {root}. "
            "Expected e.g. DSL-TL-Corpus/{lang.upper()}-DSL-TL/{prefix}_{split}.tsv. "
            "Clone https://github.com/LanguageTechnologyLab/DSL-TL"
        )
    return list(_iter_dsltl_file(path, lang, skip_ambiguous))


def label_set(examples: list[DSLExample]) -> list[str]:
    return sorted({ex.country for ex in examples if ex.country})
