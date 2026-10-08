from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

from traffic_law_assistant.config import Settings


@dataclass(frozen=True)
class ModelBundle:
    embedder: object
    reranker: object
    tokenizer: object
    language_model: object


class ModelLoader:
    """Thread-safe lazy loader that caches one model bundle per application."""

    def __init__(
        self,
        settings: Settings,
        factory: Callable[[Settings], ModelBundle] | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory or self._load_from_huggingface
        self._bundle: ModelBundle | None = None
        self._lock = RLock()

    def load(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle
        with self._lock:
            if self._bundle is None:
                self._bundle = self._factory(self._settings)
            return self._bundle

    @staticmethod
    def _load_from_huggingface(settings: Settings) -> ModelBundle:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        embedder = SentenceTransformer(settings.embedding_model_id)
        reranker = CrossEncoder(settings.reranker_model_id, max_length=512)
        tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return ModelBundle(
            embedder=embedder,
            reranker=reranker,
            tokenizer=tokenizer,
            language_model=language_model,
        )
