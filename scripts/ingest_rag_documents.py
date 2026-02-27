#!/usr/bin/env python3
"""
Ingest documents into the RAG ChromaDB vector store.

Usage:
    uv run python scripts/ingest_rag_documents.py [--dir PATH] [--config PATH]

Examples:
    uv run python scripts/ingest_rag_documents.py --dir ./knowledge_base
    uv run python scripts/ingest_rag_documents.py --dir ./docs --config conf.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from open_llm_vtuber.rag import ChromaRAG
from open_llm_vtuber.config_manager import read_yaml, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG ChromaDB vector store"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing .txt and .md files to ingest",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf.yaml",
        help="Path to config file (default: conf.yaml)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Characters per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks (default: 64)",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Error: Directory not found: {dir_path}")
        return 1

    persist_dir = "./cache/rag_chroma"
    collection_name = "open_llm_vtuber_rag"
    embedding_model = "BorisTM/bge-m3_en_ru"

    if Path(args.config).exists():
        try:
            config_data = read_yaml(args.config)
            config = validate_config(config_data)
            if config.system_config and config.system_config.rag_config:
                rc = config.system_config.rag_config
                persist_dir = rc.persist_directory
                collection_name = rc.collection_name
                embedding_model = rc.embedding_model
        except Exception as e:
            print(f"Warning: Could not load config: {e}. Using defaults.")

    print(f"Ingesting from {dir_path}...")
    print(f"Persist directory: {persist_dir}")
    print(f"Collection: {collection_name}")
    print(f"Embedding model: {embedding_model}")

    try:
        rag = ChromaRAG(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        count = rag.add_documents_from_directory(
            str(dir_path),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Successfully ingested {count} document chunks.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
