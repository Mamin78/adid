"""fastText LID and GlotLID wrappers for NADI country-level evaluation.

Both models are tiny (~1GB max) and run on CPU in milliseconds per sentence.

--- fastText lid.218 (NLLB) ---

The NLLB fasttext model covers 218 language codes (ISO 639-3 + script).
For Arabic it has these relevant codes:
    arb_Arab  Modern Standard Arabic
    arz_Arab  Egyptian Arabic            → EG
    acm_Arab  Mesopotamian (Iraqi)       → IQ
    apc_Arab  North Levantine (SY/LB)
    ajp_Arab  South Levantine (PS/JO)
    ary_Arab  Moroccan Arabic            → MA
    aeb_Arab  Tunisian Arabic            → TN
    arq_Arab  Algerian Arabic            → DZ
    shu_Arab  Chadian/Sudanese Arabic    → SD (approx)

All other countries (KW, QA, AE, BH, OM, YE, LY, MR, SO, ...) have no
dedicated code — the model returns arb_Arab for Gulf varieties. This is
precisely the gap our geo-n-gram model aims to fill.

Download (automatic on first use, or manually):
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.218.bin

--- GlotLID v3 ---

GlotLID covers 1,600+ language-script pairs. For Arabic dialects it adds:
    ajp_Arab  South Levantine
    apc_Arab  North Levantine
    ary_Arab  Moroccan
    aeb_Arab  Tunisian
    arq_Arab  Algerian

Download:
    from huggingface_hub import hf_hub_download
    hf_hub_download("cis-lmu/glotlid", filename="model.bin", local_dir="models/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

# arb_Arab / arb → no unique country assignment; omit from NADI mapping.
# For multi-country codes (apc_Arab covers SY+LB), we assign the more
# populous country. This is explicitly documented as a limitation.
_FASTTEXT_TO_NADI: dict[str, str] = {
    "arz_Arab": "EG", "arz": "EG",
    "acm_Arab": "IQ", "acm": "IQ",
    "apc_Arab": "SY", "apc": "SY",   # SY or LB; SY chosen as larger
    "ajp_Arab": "JO", "ajp": "JO",   # PS or JO; JO chosen
    "ary_Arab": "MA", "ary": "MA",
    "aeb_Arab": "TN", "aeb": "TN",
    "arq_Arab": "DZ", "arq": "DZ",
    "shu_Arab": "SD", "shu": "SD",
    # Non-dialect Arabic → no prediction (returns None)
    "arb_Arab": None, "arb": None,
    "ara_Arab": None, "ara": None,
}

_DEFAULT_FASTTEXT_PATH = Path("models/model.bin")      # facebook/fasttext-language-identification
_DEFAULT_GLOTLID_PATH = Path("models/glotlid/model.bin")  # cis-lmu/glotlid


class FasttextLID:
    """Wraps the NLLB fasttext lid.218 model for NADI country prediction.

    Args:
        model_path: Path to lid.218.bin. Downloads automatically if not found
                    and HuggingFace hub is available.
        k: Number of top predictions to consider when voting for a country.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_FASTTEXT_PATH,
        k: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        self.k = k
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import fasttext
        except ImportError as e:
            raise ImportError("pip install fasttext-wheel") from e

        if not self.model_path.exists():
            self.model_path = self._download()

        self._model = fasttext.load_model(str(self.model_path))

    def _download(self) -> Path:
        dest = _DEFAULT_FASTTEXT_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="facebook/fasttext-language-identification",
                filename="model.bin",
                local_dir=str(dest.parent),
            )
            return Path(path)
        except Exception:
            pass
        # Manual fallback URL (NLLB lid.218)
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.218.bin"
        print(f"Downloading fasttext lid.218 from {url} ...")
        urllib.request.urlretrieve(url, dest)
        return dest

    def predict_country(self, text: str) -> str | None:
        """Return a NADI ISO country code, or None if arb/MSA/no Arabic."""
        self._load()
        text_clean = text.replace("\n", " ").strip()
        labels, _ = self._model.predict(text_clean, k=self.k)
        for label in labels:
            lang = label.removeprefix("__label__")
            country = _FASTTEXT_TO_NADI.get(lang)
            if country is not None:
                return country
            if country is None and lang in _FASTTEXT_TO_NADI:
                return None   # explicitly mapped to None (MSA)
        return None

    def predict_batch(self, texts: list[str]) -> list[str | None]:
        return [self.predict_country(t) for t in texts]

    def coverage(self, countries: Sequence[str]) -> dict:
        """Report which countries this model can predict vs. cannot."""
        reachable = set(_FASTTEXT_TO_NADI[k] for k in _FASTTEXT_TO_NADI if _FASTTEXT_TO_NADI[k])
        covered = sorted(c for c in countries if c in reachable)
        uncovered = sorted(c for c in countries if c not in reachable)
        return {
            "covered": covered,
            "uncovered": uncovered,
            "coverage_pct": 100 * len(covered) / max(len(countries), 1),
        }


class GlotLID:
    """Wraps the GlotLID v3 model (fasttext format) for NADI country prediction.

    GlotLID adds a few more Arabic dialect codes than lid.218 but still cannot
    distinguish Gulf countries from each other.

    Args:
        model_path: Path to glotlid model.bin.
    """

    # GlotLID uses the same ISO 639-3 codes + script suffix.
    # Mapping is the same as FasttextLID's; GlotLID adds no new Arabic varieties
    # beyond what NLLB has for our 21-country coverage.
    _LABEL_MAP = _FASTTEXT_TO_NADI

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_GLOTLID_PATH,
        k: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        self.k = k
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import fasttext
        except ImportError as e:
            raise ImportError("pip install fasttext-wheel") from e

        if not self.model_path.exists():
            self.model_path = self._download()

        self._model = fasttext.load_model(str(self.model_path))

    def _download(self) -> Path:
        dest = _DEFAULT_GLOTLID_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            # Download into a dedicated subdir so it doesn't collide with fasttext model.bin
            path = hf_hub_download(
                repo_id="cis-lmu/glotlid",
                filename="model.bin",
                local_dir=str(dest.parent),
            )
            return Path(path)
        except Exception as e:
            raise RuntimeError(
                "Could not auto-download GlotLID. "
                "Run: python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('cis-lmu/glotlid', 'model.bin', local_dir='models/')\""
            ) from e

    def predict_country(self, text: str) -> str | None:
        self._load()
        text_clean = text.replace("\n", " ").strip()
        labels, _ = self._model.predict(text_clean, k=self.k)
        for label in labels:
            lang = label.removeprefix("__label__")
            if lang in self._LABEL_MAP:
                return self._LABEL_MAP[lang]
        return None

    def predict_batch(self, texts: list[str]) -> list[str | None]:
        return [self.predict_country(t) for t in texts]
