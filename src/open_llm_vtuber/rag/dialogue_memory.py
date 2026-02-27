"""Dialogue memory for RAG: user/assistant messages, facts, summaries, user profile."""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

MEMORY_LOG_PREFIX = "[Память]"
PREVIEW_LEN = 50

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_FACT = "fact"
ROLE_SUMMARY = "summary"
ROLE_USER_PROFILE = "user_profile"

# Roles that are never deleted by cleanup (facts and profile persist)
PERSISTENT_ROLES = {ROLE_FACT, ROLE_USER_PROFILE}


def _preview(text: str) -> str:
    """Return first 50 chars for logging."""
    s = (text or "").strip()
    if len(s) <= PREVIEW_LEN:
        return s
    return s[:PREVIEW_LEN] + "..."


def _log_save(role: str, content: str) -> None:
    """Log save to memory with role and preview."""
    logger.info(f'{MEMORY_LOG_PREFIX} Сохранено {role}: "{_preview(content)}"')


class DialogueMemory:
    """
    Vector store for dialogue memory: messages, facts, summaries, user profile.

    Uses ChromaDB with metadata (role, timestamp, history_uid) for filtering.
    Supports query by similarity, cleanup of old messages, and persistent facts/profile.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "open_llm_vtuber_dialogue_memory",
        embedding_model: str = "BorisTM/bge-m3_en_ru",
    ) -> None:
        """
        Initialize dialogue memory store.

        Args:
            persist_directory: Path to persist ChromaDB.
            collection_name: ChromaDB collection name.
            embedding_model: Sentence-transformers model for embeddings.
        """
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
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
            f"DialogueMemory initialized: persist={persist_directory}, "
            f"collection={collection_name}"
        )

    def add_item(
        self,
        role: str,
        content: str,
        history_uid: str = "",
        conf_uid: str = "",
    ) -> str:
        """
        Add an item to dialogue memory.

        Args:
            role: One of user, assistant, fact, summary, user_profile.
            content: Text content.
            history_uid: Chat history identifier.
            conf_uid: Character config identifier.

        Returns:
            Generated document ID.
        """
        if not content or not content.strip():
            return ""
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        metadata: dict[str, str | int] = {
            "role": role,
            "history_uid": history_uid or "",
            "conf_uid": conf_uid or "",
            "timestamp": int(now.timestamp()),
        }
        self._collection.add(
            documents=[content.strip()],
            ids=[doc_id],
            metadatas=[metadata],
        )
        _log_save(role, content)
        return doc_id

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        history_uid: str | None = None,
        conf_uid: str | None = None,
        roles: list[str] | None = None,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Query similar items. Returns (content, id, metadata) tuples.

        Args:
            query_text: Query string.
            n_results: Max results.
            history_uid: Filter by history (optional).
            conf_uid: Filter by character config (optional) — each character has own memory.
            roles: Filter by roles (optional).

        Returns:
            List of (content, id, metadata).
        """
        count = self._collection.count()
        if count == 0:
            return []

        where_parts: list[dict[str, Any]] = []
        if history_uid:
            where_parts.append({"history_uid": history_uid})
        if conf_uid:
            where_parts.append({"conf_uid": conf_uid})
        if roles:
            where_parts.append({"role": {"$in": roles}})

        where_filter: dict[str, Any] | None = None
        if len(where_parts) == 1:
            where_filter = where_parts[0]
        elif len(where_parts) > 1:
            where_filter = {"$and": where_parts}

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        out: list[tuple[str, str, dict[str, Any]]] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            if content:
                out.append((content, str(doc_id), meta))
        return out

    def get_all_facts(
        self, history_uid: str, conf_uid: str = "", limit: int = 50
    ) -> list[str]:
        """
        Get all facts for this history, newest first (for merge/resolve contradictions).

        Args:
            history_uid: Chat history identifier.
            conf_uid: Character config identifier (optional filter).
            limit: Max facts to return.

        Returns:
            List of fact content strings, newest first.
        """
        where_parts: list[dict[str, Any]] = [
            {"role": ROLE_FACT},
            {"history_uid": history_uid},
        ]
        if conf_uid:
            where_parts.append({"conf_uid": conf_uid})
        where_filter = {"$and": where_parts}

        results = self._collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        if not results or not results.get("ids"):
            return []

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        paired = list(zip(docs, metas))
        paired.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
        return [(p[0] or "").strip() for p in paired[:limit] if (p[0] or "").strip()]

    def set_user_profile(self, history_uid: str, conf_uid: str, content: str) -> None:
        """
        Set (replace) the user profile with merged content.
        Use after LLM merge to resolve contradictions.

        Args:
            history_uid: Chat history identifier.
            conf_uid: Character config identifier.
            content: Full merged profile text.
        """
        if not content or not content.strip():
            return
        self.add_item(
            role=ROLE_USER_PROFILE,
            content=content.strip(),
            history_uid=history_uid,
            conf_uid=conf_uid,
        )

    def add_to_user_profile(
        self,
        history_uid: str,
        conf_uid: str,
        new_fact: str,
    ) -> None:
        """
        Append a fact to the user profile. Prefer set_user_profile for merge logic.

        Args:
            history_uid: Chat history identifier.
            conf_uid: Character config identifier.
            new_fact: New fact to add (e.g. "Имя: Саша" or "Любит кофе").
        """
        if not new_fact or not new_fact.strip():
            return
        current = self.get_user_profile(history_uid, conf_uid)
        if current:
            merged = f"{current}\n- {new_fact.strip()}"
        else:
            merged = f"- {new_fact.strip()}"
        self.add_item(
            role=ROLE_USER_PROFILE,
            content=merged,
            history_uid=history_uid,
            conf_uid=conf_uid,
        )

    def get_user_profile(self, history_uid: str, conf_uid: str = "") -> str:
        """
        Get the latest user_profile content for this history.

        Args:
            history_uid: Chat history identifier.
            conf_uid: Character config identifier (optional filter).

        Returns:
            User profile text or empty string.
        """
        where_parts: list[dict[str, Any]] = [
            {"role": ROLE_USER_PROFILE},
            {"history_uid": history_uid},
        ]
        if conf_uid:
            where_parts.append({"conf_uid": conf_uid})
        where_filter = {"$and": where_parts}

        results = self._collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        if not results or not results.get("ids"):
            return ""

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            return ""

        # Sort by timestamp desc, take latest
        paired = list(zip(docs, metas))
        paired.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
        return (paired[0][0] or "").strip()

    def delete_older_than_days(
        self,
        days: int = 30,
        history_uid: str | None = None,
        exclude_roles: set[str] | None = None,
    ) -> int:
        """
        Delete items older than N days. Never deletes fact/user_profile by default.

        Args:
            days: Delete items older than this many days.
            history_uid: Limit to this history (optional).
            exclude_roles: Roles to never delete (default: fact, user_profile).

        Returns:
            Number of items deleted.
        """
        exclude = exclude_roles or PERSISTENT_ROLES
        cutoff = int((datetime.utcnow() - timedelta(days=days)).timestamp())

        where_parts: list[dict[str, Any]] = [{"timestamp": {"$lt": cutoff}}]
        if history_uid:
            where_parts.append({"history_uid": history_uid})
        where_parts.append({"role": {"$nin": list(exclude)}})
        where_filter = {"$and": where_parts}

        results = self._collection.get(where=where_filter, include=[])
        ids = results.get("ids", [])
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info(
            f"{MEMORY_LOG_PREFIX} Удалено {len(ids)} записей старше {days} дней."
        )
        return len(ids)

    def count(self) -> int:
        """Return total number of items."""
        return self._collection.count()

    def list_items(
        self,
        conf_uid: str,
        history_uid: str | None = None,
        role: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        List memory items with optional filters. For UI display.

        Args:
            conf_uid: Character config identifier (required).
            history_uid: Filter by history (optional). If None, all histories.
            role: Filter by role: fact, summary, user_profile (optional).
            limit: Max items to return.

        Returns:
            List of dicts: {id, content, role, timestamp}.
        """
        where_parts: list[dict[str, Any]] = [{"conf_uid": conf_uid}]
        if history_uid:
            where_parts.append({"history_uid": history_uid})
        if role:
            where_parts.append({"role": role})
        where_filter: dict[str, Any] = (
            where_parts[0] if len(where_parts) == 1 else {"$and": where_parts}
        )

        results = self._collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
            limit=limit,
        )
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        out: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            out.append(
                {
                    "id": str(doc_id),
                    "content": (content or "").strip(),
                    "role": meta.get("role", ""),
                    "timestamp": meta.get("timestamp", 0),
                }
            )
        # Sort by timestamp desc (newest first)
        out.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return out[:limit]

    def delete_by_ids(self, ids: list[str]) -> int:
        """
        Delete specific items by ID.

        Args:
            ids: List of document IDs.

        Returns:
            Number of items deleted.
        """
        if not ids:
            return 0
        valid = [i for i in ids if i]
        if not valid:
            return 0
        self._collection.delete(ids=valid)
        logger.info(f"{MEMORY_LOG_PREFIX} Удалено {len(valid)} записей по ID.")
        return len(valid)

    def delete_all(
        self,
        conf_uid: str,
        history_uid: str | None = None,
    ) -> int:
        """
        Delete all memory items for character (and optionally history).

        Args:
            conf_uid: Character config identifier.
            history_uid: If set, limit to this history.

        Returns:
            Number of items deleted.
        """
        where_parts: list[dict[str, Any]] = [{"conf_uid": conf_uid}]
        if history_uid:
            where_parts.append({"history_uid": history_uid})
        where_filter: dict[str, Any] = (
            where_parts[0] if len(where_parts) == 1 else {"$and": where_parts}
        )
        results = self._collection.get(where=where_filter, include=[])
        ids = results.get("ids", [])
        if not ids:
            return 0
        self._collection.delete(ids=list(ids))
        logger.info(f"{MEMORY_LOG_PREFIX} Очищено {len(ids)} записей.")
        return len(ids)
