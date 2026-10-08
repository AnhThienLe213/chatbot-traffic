import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request

from traffic_law_assistant.api import ChatRequest, app, chat, health, lifespan
from traffic_law_assistant.service import ChatResponse as ServiceChatResponse


class ApiTests(unittest.TestCase):
    def test_lifespan_builds_chatbot_once_and_clears_it_on_shutdown(self) -> None:
        chatbot = object()

        async def run_lifespan() -> None:
            with patch("traffic_law_assistant.api.build_chatbot", return_value=chatbot) as factory:
                async with lifespan(app):
                    self.assertIs(app.state.chatbot, chatbot)
                    factory.assert_called_once_with()
                self.assertIsNone(app.state.chatbot)

        asyncio.run(run_lifespan())

    def test_chat_endpoint_returns_answer_and_sources(self) -> None:
        chatbot = SimpleNamespace(
            answer=lambda question: ServiceChatResponse(
                answer=f"Trả lời: {question}", sources=("Căn cứ luật",)
            )
        )
        app.state.chatbot = chatbot
        request = Request({"type": "http", "app": app})

        response = chat(ChatRequest(question="  Câu hỏi  "), request)

        self.assertEqual(response.answer, "Trả lời: Câu hỏi")
        self.assertEqual(response.sources, ["Căn cứ luật"])

    def test_chat_endpoint_rejects_blank_question(self) -> None:
        app.state.chatbot = object()
        request = Request({"type": "http", "app": app})

        with self.assertRaises(HTTPException) as raised:
            chat(ChatRequest(question=" "), request)

        self.assertEqual(raised.exception.status_code, 422)

    def test_health_endpoint_reports_ready_chatbot(self) -> None:
        app.state.chatbot = object()

        self.assertEqual(health(Request({"type": "http", "app": app})), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
