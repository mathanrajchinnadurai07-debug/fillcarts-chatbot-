"""
app/embeddings.py
─────────────────
Embedding generation via sentence-transformers and ChromaDB vector store
interface. Provides:

  - EmbeddingService  : wraps the sentence-transformer model
  - ChromaService     : wraps a persistent ChromaDB collection
  - ingest_csv_to_chroma() : one-shot function to load training CSV into ChromaDB
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chromadb import Collection

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Embedding Service ────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Thin wrapper around a SentenceTransformer model.

    The model is loaded once at instantiation and reused for all encode calls
    to avoid repeated disk I/O and GPU/CPU initialisation overhead.
    """

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        """
        Initialise the embedding model.

        Args:
            model_name: HuggingFace model identifier, e.g. 'all-MiniLM-L6-v2'.
        """
        logger.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load embedding model %s: %s", model_name, exc)
            raise

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of strings into dense embeddings.

        Args:
            texts: Input strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If the model fails to encode.
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")

        try:
            vectors = self.model.encode(texts, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.error("Embedding encode failed: %s", exc)
            raise RuntimeError(f"Embedding encode error: {exc}") from exc

    def encode_single(self, text: str) -> list[float]:
        """
        Encode a single string into an embedding vector.

        Args:
            text: The input string.

        Returns:
            A single embedding vector as a list of floats.
        """
        return self.encode([text])[0]


# ─── ChromaDB Service ─────────────────────────────────────────────────────────

class ChromaService:
    """
    Manages a persistent ChromaDB collection for retrieval-augmented generation.

    Each document stored contains:
      - id        : unique identifier
      - document  : the raw text (training sample or knowledge chunk)
      - metadata  : dict with 'intent' and 'response' fields
      - embedding : precomputed sentence embedding
    """

    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        collection_name: str = settings.chroma_collection_name,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """
        Initialise the ChromaDB client and get/create the named collection.

        Args:
            persist_dir: Directory path for ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection.
            embedding_service: Optional pre-built EmbeddingService instance.
        """
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("Initialising ChromaDB at: %s", persist_dir)

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection: "Collection" = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.embedding_service = embedding_service or EmbeddingService()
            logger.info(
                "ChromaDB collection '%s' ready. Documents: %d",
                collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.error("ChromaDB initialisation failed: %s", exc)
            raise

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add (or upsert) documents into the ChromaDB collection.

        Uses upsert so it is safe to call multiple times on the same data
        without creating duplicates.

        Args:
            ids: Unique string identifiers for each document.
            documents: Raw text content of each document.
            metadatas: List of dicts with arbitrary metadata per document.

        Raises:
            ValueError: If the lists have mismatched lengths or are empty.
            RuntimeError: If ChromaDB upsert fails.
        """
        if not ids:
            raise ValueError("ids list cannot be empty.")
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, and metadatas must have equal lengths.")

        try:
            embeddings = self.embedding_service.encode(documents)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info("Upserted %d documents into ChromaDB.", len(ids))
        except Exception as exc:
            logger.error("ChromaDB upsert failed: %s", exc)
            raise RuntimeError(f"ChromaDB upsert error: {exc}") from exc

    def query(
        self,
        query_text: str,
        top_k: int = settings.top_k_retrieval,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most semantically similar documents.

        Args:
            query_text: The user's query or message.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: 'document', 'metadata', 'distance', 'id'.

        Raises:
            RuntimeError: If the query fails.
        """
        if not query_text.strip():
            logger.warning("ChromaDB query called with empty text; returning [].")
            return []

        try:
            query_embedding = self.embedding_service.encode_single(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(self.collection.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )

            hits: list[dict[str, Any]] = []
            if results and results.get("ids") and results["ids"][0]:
                for doc_id, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    hits.append(
                        {
                            "id": doc_id,
                            "document": doc,
                            "metadata": meta,
                            "distance": dist,
                            # Convert cosine distance → similarity score
                            "similarity": round(1.0 - dist, 4),
                        }
                    )
            return hits

        except Exception as exc:
            logger.error("ChromaDB query failed for text=%r: %s", query_text, exc)
            raise RuntimeError(f"ChromaDB query error: {exc}") from exc

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection (useful for full reindexing).

        Warning: This permanently deletes all stored documents.
        """
        logger.warning("Resetting ChromaDB collection — all documents will be deleted.")
        try:
            collection_name = self.collection.name
            self.client.delete_collection(collection_name)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' has been reset.", collection_name)
        except Exception as exc:
            logger.error("Failed to reset collection: %s", exc)
            raise


# ─── CSV Ingestion Helper ─────────────────────────────────────────────────────

def ingest_csv_to_chroma(
    csv_path: str | Path = settings.training_data_path,
    chroma_service: ChromaService | None = None,
    reset: bool = False,
) -> int:
    """
    Load training CSV data into ChromaDB.

    Reads the CSV at ``csv_path``, which must have columns:
      - text     : the raw utterance / knowledge text
      - intent   : the intent label
      - response : the templated/ideal response

    Args:
        csv_path: Path to the training CSV file.
        chroma_service: Optional pre-built ChromaService; created if None.
        reset: If True, clears the collection before ingesting.

    Returns:
        Number of documents successfully ingested.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
        RuntimeError: If ingestion fails.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    logger.info("Ingesting data from %s into ChromaDB...", csv_path)

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {exc}") from exc

    required_columns = {"text", "intent", "response"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows with nulls in critical columns
    df = df.dropna(subset=list(required_columns))
    df = df[df["text"].str.strip().astype(bool)]

    if df.empty:
        logger.warning("No valid rows found in %s after cleaning.", csv_path)
        return 0

    service = chroma_service or ChromaService()

    if reset:
        service.reset_collection()

    ids = [f"doc_{i}" for i in range(len(df))]
    documents = df["text"].tolist()
    metadatas = [
        {"intent": row["intent"], "response": row["response"]}
        for _, row in df.iterrows()
    ]

    service.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Ingested %d documents into ChromaDB.", len(ids))
    return len(ids)


# ─── Module-level singletons (lazy-initialised) ───────────────────────────────
# Other modules can import these directly:
#   from app.embeddings import embedding_service, chroma_service

_embedding_service: EmbeddingService | None = None
_chroma_service: ChromaService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the module-level EmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_chroma_service() -> ChromaService:
    """Return the module-level ChromaService singleton."""
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaService(embedding_service=get_embedding_service())
    return _chroma_service
