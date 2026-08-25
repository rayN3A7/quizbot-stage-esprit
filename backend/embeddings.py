"""
Wrapper autour du modèle d'embedding (sentence-transformers par défaut).

Isolé dans son propre module afin de pouvoir facilement remplacer
sentence-transformers par les embeddings OpenAI si besoin (cf. cahier des
charges, section 4 : "sentence-transformers, OpenAI Embeddings").
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from .config import settings


@lru_cache(maxsize=1)
def _get_model():
    """Charge le modèle sentence-transformers une seule fois (singleton)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encode une liste de textes en vecteurs numériques (embeddings)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
