"""Orchestrates the full voice turn: ASR -> LLM -> TTS."""
from __future__ import annotations
import time
from typing import Optional
from ..models.asr import ASRModel
from ..models.llm import LLMModel
from ..models.tts import TTSModel


class VoicePipeline:
    def __init__(self):
        self.history: list[dict] = []

    def turn(self, audio_bytes: Optional[bytes] = None, text: Optional[str] = None) -> dict:
        t0 = time.time()
        # 1. ASR
        if text is None:
            asr = ASRModel.transcribe(audio_bytes or b"")
            text = asr["text"]
            asr_ms = asr["latency_ms"]
        else:
            asr_ms = 0
        # 2. LLM
        llm = LLMModel.generate(text, self.history)
        reply = llm["text"]
        llm_ms = llm["latency_ms"]
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-12:]
        # 3. TTS
        tts = TTSModel.synthesize(reply)
        tts_ms = tts["latency_ms"]
        total_ms = round((time.time() - t0) * 1000, 1)
        return {
            "user_text": text,
            "reply_text": reply,
            "audio_b64": __import__("base64").b64encode(tts["audio"]).decode(),
            "mime": tts["mime"],
            "duration_s": tts["duration_s"],
            "timings": {
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
            },
            "demo": tts.get("demo", True),
        }

    def reset(self) -> None:
        self.history.clear()
