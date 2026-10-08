from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_law_assistant.config import Settings
from traffic_law_assistant.knowledge_base import KnowledgeBase
from traffic_law_assistant.models import ModelBundle


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float


class HybridRetriever:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        models: ModelBundle,
        settings: Settings,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._embedder = models.embedder
        self._reranker = models.reranker
        self._candidate_k = settings.candidate_k

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        candidate_count = min(self._candidate_k, len(self._knowledge_base.chunks))
        bm25_scores = self._knowledge_base.bm25.get_scores(normalized_query.lower().split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_count].tolist()

        query_embedding = self._embedder.encode(
            [normalized_query], normalize_embeddings=True
        )
        _, faiss_indices = self._knowledge_base.index.search(
            np.asarray(query_embedding, dtype=np.float32), candidate_count
        )

        candidate_indices = list(dict.fromkeys(
            bm25_indices + [int(index) for index in faiss_indices[0] if index >= 0]
        ))
        candidate_texts = [self._knowledge_base.chunks[index] for index in candidate_indices]
        pairs = [[normalized_query, text] for text in candidate_texts]
        scores = self._reranker.predict(pairs)
        ranked = sorted(
            zip(candidate_texts, scores),
            key=lambda candidate: float(candidate[1]),
            reverse=True,
        )
        return [
            RetrievedChunk(text=text, score=float(score))
            for text, score in ranked[:top_k]
        ]
