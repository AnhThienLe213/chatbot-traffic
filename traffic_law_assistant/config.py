"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "vector_db"
    )
    embedding_model_id: str = "BAAI/bge-m3"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    llm_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    top_k: int = 3
    candidate_k: int = 15
    max_new_tokens: int = 512

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            artifact_dir=Path(os.getenv("TRAFFIC_CHATBOT_ARTIFACT_DIR", defaults.artifact_dir)),
            embedding_model_id=os.getenv(
                "TRAFFIC_CHATBOT_EMBEDDING_MODEL", defaults.embedding_model_id
            ),
            reranker_model_id=os.getenv(
                "TRAFFIC_CHATBOT_RERANKER_MODEL", defaults.reranker_model_id
            ),
            llm_model_id=os.getenv("TRAFFIC_CHATBOT_LLM_MODEL", defaults.llm_model_id),
            top_k=_positive_int("TRAFFIC_CHATBOT_TOP_K", defaults.top_k),
            candidate_k=_positive_int(
                "TRAFFIC_CHATBOT_CANDIDATE_K", defaults.candidate_k
            ),
            max_new_tokens=_positive_int(
                "TRAFFIC_CHATBOT_MAX_NEW_TOKENS", defaults.max_new_tokens
            ),
        )


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer; received {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return parsed
