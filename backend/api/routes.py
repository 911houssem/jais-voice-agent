from __future__ import annotations
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from .schemas import ChatRequest, ChatResponse, TTSRequest, TurnResponse, StatusResponse
from ..config import settings
from ..models.asr import ASRModel
from ..models.llm import LLMModel
from ..models.tts import TTSModel
from ..pipeline.orchestrator import VoicePipeline

router = APIRouter()
_pipeline = VoicePipeline()


@router.get("/health")
def health():
    return {"status": "ok", "mode": settings.mode}


@router.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        mode=settings.mode,
        models={
            "asr": ASRModel.status(),
            "llm": LLMModel.status(),
            "tts": TTSModel.status(),
        },
        max_concurrent_calls=settings.max_concurrent_calls,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    res = LLMModel.generate(req.text, req.history)
    return ChatResponse(reply=res["text"], latency_ms=res["latency_ms"], demo=res.get("demo", True))


@router.post("/tts")
def tts(req: TTSRequest):
    res = TTSModel.synthesize(req.text)
    return {"audio_b64": base64.b64encode(res["audio"]).decode(),
            "mime": res["mime"], "duration_s": res["duration_s"],
            "latency_ms": res["latency_ms"], "demo": res.get("demo", True)}


@router.post("/asr", response_model=dict)
async def asr(file: UploadFile = File(...)):
    data = await file.read()
    res = ASRModel.transcribe(data)
    return res


@router.post("/turn", response_model=TurnResponse)
def turn(req: ChatRequest):
    # Text-only turn (no audio) for the demo UI; audio returned from TTS.
    p = _pipeline
    result = p.turn(text=req.text)
    return TurnResponse(**result)


@router.post("/turn/reset")
def reset():
    _pipeline.reset()
    return {"ok": True}
