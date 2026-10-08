"""Load and validate retrieval artifacts for the traffic-law knowledge base."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnowledgeBase:
    index: Any
    bm25: Any
    chunks: tuple[str, ...]


class KnowledgeBaseLoader:
    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir

    def load(self) -> KnowledgeBase:
        import faiss

        index_path = self._artifact_dir / "faiss_bge_m3.index"
        bm25_path = self._artifact_dir / "bm25_model.pkl"
        chunks_path = self._artifact_dir / "chunk_data.json"
        for path in (index_path, bm25_path, chunks_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required chatbot artifact was not found: {path}")

        index = faiss.read_index(str(index_path))
        with bm25_path.open("rb") as artifact_file:
            bm25 = pickle.load(artifact_file)
        with chunks_path.open("r", encoding="utf-8") as artifact_file:
            chunk_data = json.load(artifact_file)

        chunks = chunk_data.get("chunk_texts_with_meta")
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError(
                f"{chunks_path} must contain a string list named 'chunk_texts_with_meta'"
            )
        if not chunks:
            raise ValueError(f"{chunks_path} contains no searchable chunks")
        if len(chunks) != int(index.ntotal):
            raise ValueError(
                f"FAISS index contains {index.ntotal} vectors but {len(chunks)} chunks were loaded"
            )
        bm25_size = len(bm25.get_scores([]))
        if bm25_size != len(chunks):
            raise ValueError(
                f"BM25 index contains {bm25_size} entries but {len(chunks)} chunks were loaded"
            )
        return KnowledgeBase(index=index, bm25=bm25, chunks=tuple(chunks))
