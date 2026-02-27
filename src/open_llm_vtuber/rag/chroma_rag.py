"""ChromaDB-based RAG (Retrieval-Augmented Generation) implementation."""

import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger


class ChromaRAG:
    """
    RAG implementation using ChromaDB as the vector store.

    Supports adding documents, querying by semantic similarity, and persisting
    the database to disk. Uses sentence-transformers for embeddings.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "open_llm_vtuber_rag",
        embedding_model: str = "BorisTM/bge-m3_en_ru",
    ) -> None:
        """
        Initialize the ChromaDB RAG client.

        Args:
            persist_directory: Path to persist the ChromaDB database.
            collection_name: Name of the ChromaDB collection.
            embedding_model: Sentence-transformers model for embeddings.
        """
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaRAG initialized: persist={persist_directory}, "
            f"collection={collection_name}, embedding={embedding_model}"
        )

    def add_documents(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of text chunks to add.
            ids: Optional list of unique IDs. Auto-generated if not provided.
            metadatas: Optional list of metadata dicts per document.

        Returns:
            Number of documents added.
        """
        if not documents:
            return 0
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if len(ids) != len(documents) or len(metadatas) != len(documents):
            raise ValueError("documents, ids, and metadatas must have same length")
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Added {len(documents)} documents to RAG collection.")
        return len(documents)

    def add_documents_from_directory(
        self,
        directory: str,
        extensions: tuple[str, ...] = (".txt", ".md"),
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """
        Load text files from a directory and add them to the vector store.

        Args:
            directory: Path to the directory containing text files.
            extensions: File extensions to include.
            chunk_size: Approximate characters per chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            Total number of chunks added.
        """
        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        all_chunks: list[str] = []
        all_metadatas: list[dict[str, Any]] = []
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() not in extensions:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue
            chunks = self._chunk_text(
                text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": str(file_path), "chunk_index": i})
        if not all_chunks:
            logger.warning(f"No text content found in {directory}")
            return 0
        return self.add_documents(documents=all_chunks, metadatas=all_metadatas)

    def _chunk_text(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 64
    ) -> list[str]:
        """Split text into overlapping chunks by character count."""
        text = text.strip()
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            if start >= len(text):
                break
        return chunks

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: The query string.
            n_results: Maximum number of results to return.
            where: Optional metadata filter.

        Returns:
            List of document text strings, most similar first.
        """
        count = self._collection.count()
        if count == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, count),
        }
        if where is not None:
            kwargs["where"] = where
        if kwargs["n_results"] <= 0:
            return []
        results = self._collection.query(**kwargs)
        documents = results.get("documents", [[]])
        if not documents or not documents[0]:
            return []
        return list(documents[0])

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
