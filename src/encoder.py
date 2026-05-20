import re
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-loaded singleton so the model is only downloaded/loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def encode_prompt(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("Cannot encode an empty string.")
    return _get_model().encode(clean_text(text))
