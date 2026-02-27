"""RAG (Retrieval-Augmented Generation) configuration."""

from pydantic import Field
from typing import ClassVar, Dict

from .i18n import I18nMixin, Description


class RAGConfig(I18nMixin):
    """Configuration for RAG with ChromaDB vector store."""

    enabled: bool = Field(False, alias="enabled")
    persist_directory: str = Field("./cache/rag_chroma", alias="persist_directory")
    collection_name: str = Field("open_llm_vtuber_rag", alias="collection_name")
    embedding_model: str = Field("BorisTM/bge-m3_en_ru", alias="embedding_model")
    n_results: int = Field(5, alias="n_results")
    documents_dir: str | None = Field(
        None,
        alias="documents_dir",
        description="Directory to load documents from at startup (optional)",
    )
    dialogue_memory_collection: str = Field(
        "open_llm_vtuber_dialogue_memory",
        alias="dialogue_memory_collection",
        description="ChromaDB collection for dialogue memory (messages, facts, profile)",
    )
    memory_n_results: int = Field(
        5,
        alias="memory_n_results",
        description="Number of similar messages to retrieve for context",
    )
    memory_cleanup_days: int = Field(
        30,
        alias="memory_cleanup_days",
        description="Delete old messages (except facts/profile) after this many days",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable RAG to augment LLM with knowledge base context",
            zh="启用 RAG 以用知识库上下文增强 LLM",
        ),
        "persist_directory": Description(
            en="Path to persist ChromaDB vector store",
            zh="ChromaDB 向量存储持久化路径",
        ),
        "collection_name": Description(
            en="ChromaDB collection name",
            zh="ChromaDB 集合名称",
        ),
        "embedding_model": Description(
            en="Sentence-transformers model for embeddings (e.g. BorisTM/bge-m3_en_ru)",
            zh="用于嵌入的 sentence-transformers 模型",
        ),
        "n_results": Description(
            en="Number of document chunks to retrieve per query",
            zh="每次查询检索的文档块数量",
        ),
        "documents_dir": Description(
            en="Optional directory to auto-load documents from at startup",
            zh="启动时自动加载文档的可选目录",
        ),
    }
