"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

PREPROC_VERSION = "1.2.0"


import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove Tatweel / Kashida
    text = text.replace("ـ", "")

    # 3. Reduce repeated characters / punctuation to at most 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 4. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    # TODO(Lab 1): replace supported PII with <PHONE> / <NATIONAL_ID>.
    raise NotImplementedError("Implement mask_pii() in Lab 1")


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    # TODO(Lab 1): compose masking and normalisation in the intended order.
    raise NotImplementedError("Implement preprocess() in Lab 1")
