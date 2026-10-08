from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.service import TrafficChatbot, build_chatbot


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    application.state.chatbot = await run_in_threadpool(build_chatbot)
    try:
        yield
    finally:
        application.state.chatbot = None


app = FastAPI(
    title="Traffic Law Assistant API",
    description="Hỏi đáp pháp luật giao thông đường bộ Việt Nam.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
def health(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "chatbot", None) is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready")
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    chatbot: TrafficChatbot = request.app.state.chatbot
    response = chatbot.answer(question)
    return ChatResponse(answer=response.answer, sources=list(response.sources))
