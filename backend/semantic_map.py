"""
Carte sémantique d'un document.

Projette en 2D les embeddings des chunks indexés, afin de visualiser la
structure du cours : les passages traitant d'un même concept se retrouvent
proches, les sections distinctes forment des groupes séparés.

Deux usages :
  - pour l'enseignant, voir comment son support se découpe conceptuellement et
    quelles régions les quiz interrogent réellement ;
  - superposer les résultats des étudiants pour faire apparaître les régions du
    cours mal comprises (une zone de l'espace où les scores s'effondrent
    correspond à un concept à retravailler).

Réduction de dimension : ACP calculée par SVD avec numpy uniquement, pour ne
pas ajouter de dépendance (scikit-learn / UMAP) à un projet qui tourne déjà sur
une machine contrainte. Si scikit-learn est présent, t-SNE est utilisé à la
place car il sépare bien mieux les groupes ; sinon on retombe sur l'ACP.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .vectorstore import get_vector_store


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """ACP vers 2 dimensions via SVD (numpy seul).

    On centre les données puis on projette sur les deux premières composantes
    principales — les directions de plus grande variance, c'est-à-dire les deux
    axes selon lesquels les passages diffèrent le plus.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # full_matrices=False : on n'a besoin que des premières composantes.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _project(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    """Projette en 2D, en préférant t-SNE si scikit-learn est disponible."""
    n = matrix.shape[0]
    if n < 3:
        return _pca_2d(matrix) if n > 1 else np.zeros((n, 2)), "acp"

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return _pca_2d(matrix), "acp"

    # perplexity doit rester strictement inférieure au nombre d'échantillons.
    perplexity = max(2.0, min(30.0, (n - 1) / 3.0))
    try:
        coords = TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=42,
        ).fit_transform(matrix)
        return np.asarray(coords), "t-sne"
    except Exception:
        # t-SNE est capricieux sur les très petits jeux : repli silencieux.
        return _pca_2d(matrix), "acp"


def _normalize_to_unit_square(coords: np.ndarray) -> np.ndarray:
    """Ramène les coordonnées dans [0,1]² pour un affichage indépendant de l'échelle."""
    if coords.size == 0:
        return coords
    mins = coords.min(axis=0, keepdims=True)
    spans = coords.max(axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0     # évite la division par zéro si tous alignés
    return (coords - mins) / spans


def build_semantic_map(document_id: str, max_chunks: int = 300) -> dict:
    """Construit la carte sémantique d'un document indexé.

    Retourne les points projetés avec, pour chacun, un aperçu du texte et sa
    page d'origine. Lève ValueError si le document n'est pas indexé.
    """
    store = get_vector_store()
    payload = store.raw_chunks(document_id, limit=max_chunks)

    embeddings = payload.get("embeddings") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    if not embeddings:
        raise ValueError(
            "Aucun chunk indexé pour ce document (ou embeddings indisponibles). "
            "Téléversez-le à nouveau pour reconstruire l'index."
        )

    matrix = np.asarray(embeddings, dtype=float)
    coords, method = _project(matrix)
    coords = _normalize_to_unit_square(np.asarray(coords, dtype=float))

    points: List[dict] = []
    for i, (x, y) in enumerate(coords):
        text = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        preview = " ".join(str(text).split())[:160]
        points.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "preview": preview,
            "page": meta.get("source_page"),
            "chunk_index": meta.get("chunk_index", i),
        })

    return {
        "document_id": document_id,
        "method": method,
        "num_points": len(points),
        "points": points,
    }


def overlay_performance(semantic_map: dict, results: List[dict]) -> dict:
    """Superpose les résultats des étudiants sur la carte.

    Chaque question porte un `source_excerpt` ; on rattache la question au
    chunk dont l'aperçu partage le plus de mots avec cet extrait, puis on
    agrège les scores obtenus par les étudiants sur ce chunk.

    Un chunk sans question rattachée reste neutre (score None) : il n'a jamais
    été évalué, ce qui est une information différente d'un mauvais score.
    """
    buckets: dict[int, list[float]] = {}

    previews = [
        (p["chunk_index"], set(p["preview"].lower().split()))
        for p in semantic_map["points"]
    ]

    for result in results:
        for graded in result.get("graded_answers", []):
            excerpt = str(graded.get("source_excerpt") or graded.get("question") or "")
            words = set(excerpt.lower().split())
            if not words:
                continue
            best_idx, best_overlap = None, 0
            for chunk_index, preview_words in previews:
                overlap = len(words & preview_words)
                if overlap > best_overlap:
                    best_idx, best_overlap = chunk_index, overlap
            # Un recouvrement d'un seul mot est du bruit : on exige au moins 3.
            if best_idx is not None and best_overlap >= 3:
                buckets.setdefault(best_idx, []).append(float(graded.get("score", 0.0)))

    for point in semantic_map["points"]:
        scores = buckets.get(point["chunk_index"])
        point["score"] = round(sum(scores) / len(scores), 3) if scores else None
        point["attempts"] = len(scores) if scores else 0

    semantic_map["covered_points"] = sum(
        1 for p in semantic_map["points"] if p["attempts"] > 0
    )
    return semantic_map
