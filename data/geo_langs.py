"""Central registry of supported languages and their geographic TLD mappings.

Each LanguageConfig defines:
  - oscar_code      : language code for OSCAR-2201 streaming (ISO 639-1)
  - tld_map         : ccTLD → ISO-2 country code (for CC-TLD filtering)
  - dsl_prefix      : label prefix used in DSLCC (e.g. "es-" → "es-MX")
  - notes           : known noise sources for this language's TLD filtering

Noise caveats:
  - French .ca / .be are bilingual; French-Canadian is a minority of .ca content.
  - Spanish .ar collides visually with Arabic ISO code but is Argentina's TLD.
  - Portuguese .br is by far the dominant signal (Brazil >> Portugal online).
  - Arabic IDN TLDs (e.g. .مصر) are added but rarely seen in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    name: str
    oscar_code: str           # HuggingFace OSCAR-2201 language code
    tld_map: dict[str, str]   # ccTLD (lowercase) → ISO-2 country code
    dsl_prefix: str = ""      # prefix in DSLCC labels, e.g. "es-"
    notes: str = ""

    def country_for_tld(self, tld: str) -> str | None:
        return self.tld_map.get(tld.lower())

    def countries(self) -> list[str]:
        return sorted(set(self.tld_map.values()))


# ── Arabic ────────────────────────────────────────────────────────────────────
ARABIC = LanguageConfig(
    name="Arabic",
    oscar_code="ar",
    dsl_prefix="ar-",
    tld_map={
        "dz": "DZ", "bh": "BH", "km": "KM", "dj": "DJ",
        "eg": "EG", "iq": "IQ", "jo": "JO", "kw": "KW",
        "lb": "LB", "ly": "LY", "mr": "MR", "ma": "MA",
        "om": "OM", "ps": "PS", "qa": "QA", "sa": "SA",
        "so": "SO", "sd": "SD", "sy": "SY", "tn": "TN",
        "ae": "AE", "ye": "YE",
        # Arabic IDN TLDs (rare in practice)
        "مصر": "EG", "السعودية": "SA", "الإمارات": "AE",
        "قطر": "QA", "البحرين": "BH", "الجزائر": "DZ",
        "المغرب": "MA", "تونس": "TN", "اليمن": "YE",
        "ليبيا": "LY", "فلسطين": "PS", "الأردن": "JO",
        "عمان": "OM",
    },
    notes="Largest and cleanest TLD coverage of the three languages.",
)

# ── Spanish ───────────────────────────────────────────────────────────────────
SPANISH = LanguageConfig(
    name="Spanish",
    oscar_code="es",
    dsl_prefix="es-",
    tld_map={
        "mx": "MX",  # Mexico
        "ar": "AR",  # Argentina  (note: "ar" ≠ Arabic ISO code — TLD vs ISO)
        "co": "CO",  # Colombia
        "pe": "PE",  # Peru
        "cl": "CL",  # Chile
        "ve": "VE",  # Venezuela
        "ec": "EC",  # Ecuador
        "gt": "GT",  # Guatemala
        "cu": "CU",  # Cuba
        "bo": "BO",  # Bolivia
        "do": "DO",  # Dominican Republic
        "hn": "HN",  # Honduras
        "py": "PY",  # Paraguay
        "sv": "SV",  # El Salvador
        "ni": "NI",  # Nicaragua
        "cr": "CR",  # Costa Rica
        "pa": "PA",  # Panama
        "uy": "UY",  # Uruguay
        "es": "ES",  # Spain
        # Puerto Rico: mostly .com; skip .pr (almost no domains)
    },
    notes=(
        "Large online presence for MX/AR/ES; smaller for Central America. "
        ".ar TLD is Argentina (not the Arabic ISO code — context is always language-filtered). "
        "PR uses .com almost exclusively."
    ),
)

# ── Portuguese ────────────────────────────────────────────────────────────────
PORTUGUESE = LanguageConfig(
    name="Portuguese",
    oscar_code="pt",
    dsl_prefix="pt-",
    tld_map={
        "br": "BR",  # Brazil (overwhelmingly dominant — ~95% of pt web)
        "pt": "PT",  # Portugal
        "ao": "AO",  # Angola
        "mz": "MZ",  # Mozambique
        "cv": "CV",  # Cape Verde
        "gw": "GW",  # Guinea-Bissau
        "st": "ST",  # São Tomé and Príncipe
        "tl": "TL",  # Timor-Leste
    },
    notes=(
        "BR dominates overwhelmingly (~95%). "
        "AO/MZ have some presence; others (GW/ST/TL) are extremely sparse. "
        "BR vs PT is the canonical 2-class evaluation task."
    ),
)

# ── French ────────────────────────────────────────────────────────────────────
FRENCH = LanguageConfig(
    name="French",
    oscar_code="fr",
    dsl_prefix="fr-",
    tld_map={
        "fr": "FR",   # France (dominant)
        "be": "BE",   # Belgium (Dutch/French bilingual — noisy)
        "ca": "CA",   # Canada (English/French bilingual — noisy)
        "sn": "SN",   # Senegal
        "ci": "CI",   # Ivory Coast
        "cm": "CM",   # Cameroon
        "cd": "CD",   # DR Congo
        "mg": "MG",   # Madagascar
        "tn": "TN",   # Tunisia (Arabic/French)
        "ma": "MA",   # Morocco (Arabic/French)
        "dz": "DZ",   # Algeria (Arabic/French)
        "lu": "LU",   # Luxembourg (French/German/Luxembourgish)
        "ch": "CH",   # Switzerland (quadrilingual — noisy)
    },
    notes=(
        "FR is clean; BE/CA/CH are bilingual and noisy. "
        "African TLDs (SN/CI/CM) have low volume. "
        "TN/MA/DZ are Arabic-dominant; French content is a minority. "
        "Use with caution — less suitable than Arabic or Spanish."
    ),
)


# ── English ───────────────────────────────────────────────────────────────────
ENGLISH = LanguageConfig(
    name="English",
    oscar_code="en",
    dsl_prefix="en-",
    tld_map={
        "us": "US",   # United States (rare — most .com is used instead)
        "uk": "GB",   # United Kingdom (.uk is legacy; .co.uk is common)
        "au": "AU",   # Australia
        "nz": "NZ",   # New Zealand
        "ca": "CA",   # Canada (English + French)
        "ie": "IE",   # Ireland
        "za": "ZA",   # South Africa
        "in": "IN",   # India (English is official)
        "ng": "NG",   # Nigeria
        "gh": "GH",   # Ghana
        "ke": "KE",   # Kenya
    },
    notes=(
        "US uses .com almost exclusively; .us is very rare. "
        ".uk and .co.uk both used in Britain. "
        ".ca is bilingual (English + French). "
        "Primarily useful for US vs GB distinction (DSL-TL benchmark)."
    ),
)


# ── German ────────────────────────────────────────────────────────────────────
GERMAN = LanguageConfig(
    name="German",
    oscar_code="de",
    dsl_prefix="de-",
    tld_map={
        "de": "DE",   # Germany (dominant)
        "at": "AT",   # Austria
        "ch": "CH",   # Switzerland (quadrilingual — noisy but German-dominant)
        "li": "LI",   # Liechtenstein (very small, expect sparse data)
        "lu": "LU",   # Luxembourg (French/German/Luxembourgish trilingual)
    },
    notes=(
        "DE dominates; AT has reasonable coverage. "
        "CH is quadrilingual (DE/FR/IT/RM); German is the plurality but noisy. "
        "LI and LU will be very sparse."
    ),
)

# ── Russian ────────────────────────────────────────────────────────────────────
RUSSIAN = LanguageConfig(
    name="Russian",
    oscar_code="ru",
    dsl_prefix="ru-",
    tld_map={
        "ru": "RU",   # Russia (overwhelmingly dominant)
        "ua": "UA",   # Ukraine (Russian is widely used alongside Ukrainian)
        "by": "BY",   # Belarus (Russian is co-official)
        "kz": "KZ",   # Kazakhstan (Russian is co-official, significant web presence)
        "kg": "KG",   # Kyrgyzstan (Russian widely used)
        "md": "MD",   # Moldova (Russian widely used)
        "am": "AM",   # Armenia (Russian widely used)
        "ge": "GE",   # Georgia (Russian widely used, historic)
        "az": "AZ",   # Azerbaijan (Russian widely used)
    },
    notes=(
        "RU dominates overwhelmingly. "
        "Post-Soviet states (UA/BY/KZ/KG) have significant Russian-language web presence "
        "but Russian may not be the native variety — L2 influence possible. "
        "UA content may shift toward Ukrainian post-2022."
    ),
)


# Registry for lookup by Oscar code or name
LANGUAGES: dict[str, LanguageConfig] = {
    "ar": ARABIC,   "arabic":     ARABIC,
    "es": SPANISH,  "spanish":    SPANISH,
    "pt": PORTUGUESE, "portuguese": PORTUGUESE,
    "fr": FRENCH,   "french":     FRENCH,
    "en": ENGLISH,  "english":    ENGLISH,
    "de": GERMAN,   "german":     GERMAN,
    "ru": RUSSIAN,  "russian":    RUSSIAN,
}


def get(code_or_name: str) -> LanguageConfig:
    """Look up a LanguageConfig by Oscar code or English name (case-insensitive)."""
    key = code_or_name.lower()
    if key not in LANGUAGES:
        raise ValueError(
            f"Unknown language {code_or_name!r}. "
            f"Supported: {sorted(set(LANGUAGES.keys()))}"
        )
    return LANGUAGES[key]
