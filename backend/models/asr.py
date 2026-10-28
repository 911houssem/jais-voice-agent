"""Jais-ASR wrapper (speech -> text). Falls back to a demo stub when unavailable."""
from __future__ import annotations
import time
from typing import Optional
from ..config import settings

try:
    import torch
    _TORCH = True
except Exception:
    _TORCH = False

_MODEL = None
_LOADED = False
_LOAD_ERROR: Optional[str] = None


class ASRModel:
    name = "Jais-ASR"
    languages = ["ar", "arb", "arz", "en", "fr", "es"]

    @classmethod
    def load(cls) -> None:
        global _MODEL, _LOADED, _LOAD_ERROR
        if _LOADED:
            return
        if settings.is_demo:
            _LOAD_ERROR = "demo mode"
            _LOADED = True
            return
        try:
            # Jais-ASR ships custom code; load via transformers AutoModel if present
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            ref = settings.ref(settings.asr_model_path)
            tok = settings.hf_token or None
            dtype = settings.asr_torch_dtype
            _MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
                ref, trust_remote_code=True, torch_dtype=dtype,
                device_map=settings.device, token=tok,
            )
            cls._processor = AutoProcessor.from_pretrained(ref, trust_remote_code=True, token=tok)
            _LOADED = True
        except Exception as e:
            _LOAD_ERROR = f"{type(e).__name__}: {e}"
            _LOADED = True

    @classmethod
    def transcribe(cls, audio_bytes: bytes) -> dict:
        cls.load()
        t0 = time.time()
        if settings.is_demo or _MODEL is None:
            # Demo: deterministic mock transcription
            text = "مرحباً، أريد حجز موعد في العيادة."
            return {
                "text": text,
                "language": "arb",
                "latency_ms": round((time.time() - t0) * 1000 + 180, 1),
                "demo": True,
            }
        # Real path
        import io, soundfile as sf, numpy as np
        data, sr = sf.read(io.BytesIO(audio_bytes))
        # Jais-ASR (Whisper-based) expects 16 kHz mono
        if sr != 16000 and len(data) > 0:
            n_out = int(len(data) * 16000 / sr)
            x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
            sr = 16000
        if data.ndim > 1:
            data = data.mean(axis=1)
        inputs = cls._processor(data, sampling_rate=sr, return_tensors="pt").to(settings.device)
        with torch.no_grad():
            ids = _MODEL.generate(**inputs)
        text = cls._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return {"text": text, "language": "arb",
                "latency_ms": round((time.time() - t0) * 1000, 1), "demo": False}

    @classmethod
    def status(cls) -> dict:
        cls.load()
        return {
            "name": cls.name,
            "loaded": _MODEL is not None,
            "demo": settings.is_demo or _MODEL is None,
            "error": _LOAD_ERROR,
            "path": str(settings.asr_path),
        }
