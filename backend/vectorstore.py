"""
Base de données vectorielle (ChromaDB) pour le stockage et le retrieval
des chunks de cours indexés, au coeur du pipeline RAG.
"""
from __future__ import annotations

from typing import List

from .config import settings
from .embeddings import embed_texts
from .ingestion import Chunk


class VectorStore:
    """Fine couche au-dessus de ChromaDB, une collection par document."""

    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

    def _collection_name(self, document_id: str) -> str:
        return f"doc_{document_id}"

    def index_document(self, document_id: str, chunks: List[Chunk]) -> int:
        """Vectorise et stocke les chunks d'un document. Retourne le nombre indexé."""
        if not chunks:
            return 0

        collection = self._client.get_or_create_collection(
            name=self._collection_name(document_id),
            metadata={"hnsw:space": "cosine"},
        )
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
        metadatas = [{"chunk_index": c.chunk_index, "source_page": c.source_page or 0} for c in chunks]

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(chunks)

    def query(self, document_id: str, query_text: str, top_k: int | None = None) -> List[dict]:
        """Retourne les k passages les plus pertinents pour une requête sémantique.

        Les embeddings sont inclus dans le résultat : la sélection MMR
        (cf. quiz_generator._mmr_select) a besoin de comparer les passages
        entre eux, pas seulement à la requête.
        """
        top_k = top_k or settings.TOP_K_CHUNKS
        collection = self._client.get_or_create_collection(
            name=self._collection_name(document_id),
            metadata={"hnsw:space": "cosine"},
        )
        query_embedding = embed_texts([query_text])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings") or [[]]
        embeddings = embeddings[0] if embeddings else []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            emb = embeddings[i] if i < len(embeddings) else None
            hits.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "embedding": list(emb) if emb is not None else None,
            })
        return hits

    def all_texts(self, document_id: str, limit: int = 400) -> List[str]:
        """Retourne le texte des chunks indexés d'un document.

        Sert à caractériser le document lui-même (extraction de mots-clés
        discriminants) plutôt que de l'interroger avec une requête générique
        identique pour tous les cours.
        """
        try:
            collection = self._client.get_collection(name=self._collection_name(document_id))
        except Exception:
            return []
        try:
            data = collection.get(limit=limit, include=["documents"])
        except Exception:
            return []
        return [d for d in (data.get("documents") or []) if d]

    def raw_chunks(self, document_id: str, limit: int = 300) -> dict:
        """Retourne les chunks bruts d'un document avec leurs embeddings.

        Utilisé par la carte sémantique (semantic_map.py), qui a besoin des
        vecteurs eux-mêmes pour les projeter en 2D — contrairement à query()
        qui répond à une requête donnée.
        """
        empty = {"documents": [], "metadatas": [], "embeddings": []}
        try:
            collection = self._client.get_collection(name=self._collection_name(document_id))
        except Exception:
            return empty
        try:
            data = collection.get(
                limit=limit, include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return empty
        return {
            "documents": data.get("documents") or [],
            "metadatas": data.get("metadatas") or [],
            "embeddings": [list(e) for e in (data.get("embeddings") or [])],
        }

    def document_exists(self, document_id: str) -> bool:
        existing = [c.name for c in self._client.list_collections()]
        return self._collection_name(document_id) in existing

    def chunk_count(self, document_id: str) -> int:
        try:
            collection = self._client.get_collection(name=self._collection_name(document_id))
            return collection.count()
        except Exception:
            return 0


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Singleton simple pour éviter de rouvrir le client Chroma à chaque appel."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
