from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str
    latency_ms: float
    demo: bool


class TTSRequest(BaseModel):
    text: str


class ASRResponse(BaseModel):
    text: str
    language: str
    latency_ms: float
    demo: bool


class TurnResponse(BaseModel):
    user_text: str
    reply_text: str
    audio_b64: str
    mime: str
    duration_s: float
    timings: dict
    demo: bool


class StatusResponse(BaseModel):
    mode: str
    models: dict
    max_concurrent_calls: int
