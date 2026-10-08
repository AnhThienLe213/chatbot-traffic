from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.knowledge_base import KnowledgeBaseLoader
from src.models import ModelBundle, ModelLoader
from src.retrieval import HybridRetriever

SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo chuyên tư vấn pháp luật giao thông đường bộ Việt Nam. "
    "Hãy dựa trên văn bản luật được cung cấp để trả lời chính xác, ngắn gọn và lịch sự. "
    "Nếu tài liệu không đủ căn cứ, hãy nói rõ điều đó và không tự suy đoán."
)


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    sources: tuple[str, ...]


class TrafficChatbot:
    def __init__(
        self,
        models: ModelBundle,
        retriever: HybridRetriever,
        settings: Settings,
    ) -> None:
        self._models = models
        self._retriever = retriever
        self._settings = settings

    def answer(self, question: str) -> ChatResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")

        retrieved = self._retriever.search(normalized_question, self._settings.top_k)
        sources = tuple(chunk.text for chunk in retrieved)
        context = "\n\n".join(sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"VĂN BẢN LUẬT:\n{context}\n\nCÂU HỎI:\n{normalized_question}",
            },
        ]
        tokenizer = self._models.tokenizer
        language_model = self._models.language_model
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to(language_model.device)

        import torch

        with torch.inference_mode():
            generated_ids = language_model.generate(
                **model_inputs,
                max_new_tokens=self._settings.max_new_tokens,
                temperature=0.3,
                do_sample=True,
            )
        answer_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return ChatResponse(answer=answer, sources=sources)


def build_chatbot(settings: Settings | None = None) -> TrafficChatbot:
    app_settings = settings or Settings.from_env()
    knowledge_base = KnowledgeBaseLoader(app_settings.artifact_dir).load()
    models = ModelLoader(app_settings).load()
    retriever = HybridRetriever(knowledge_base, models, app_settings)
    return TrafficChatbot(models, retriever, app_settings)
