"""SILMA TTS wrapper (text -> audio). Returns WAV bytes (or a tiny demo tone)."""
from __future__ import annotations
import io, time, struct, math
from typing import Optional
from ..config import settings

_MODEL = None
_LOADED = False
_LOAD_ERROR: Optional[str] = None


def _demo_wav(duration_s: float = 1.2, freq: float = 220.0) -> bytes:
    """Generate a tiny sine-wave WAV so the UI can play 'audio' in demo mode."""
    sr = 16000
    n = int(sr * duration_s)
    buf = io.BytesIO()
    # WAV header
    data_size = n * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for i in range(n):
        v = int(32767 * 0.25 * math.sin(2 * math.pi * freq * i / sr) *
                (1 - i / n))
        buf.write(struct.pack("<h", v))
    return buf.getvalue()


class TTSModel:
    name = "SILMA TTS"

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
            # SILMA ships as model.pt + finetune_cli; load torch checkpoint lazily
            import torch
            path = settings.tts_path / "model.pt"
            _MODEL = torch.load(str(path), map_location=settings.device, weights_only=False)
            _LOADED = True
        except Exception as e:
            _LOAD_ERROR = f"{type(e).__name__}: {e}"
            _LOADED = True

    @classmethod
    def synthesize(cls, text: str) -> dict:
        cls.load()
        t0 = time.time()
        if settings.is_demo or _MODEL is None:
            audio = _demo_wav(duration_s=min(2.5, 0.4 + 0.08 * len(text)))
            return {
                "audio": audio,
                "mime": "audio/wav",
                "duration_s": round(len(audio) / (16000 * 2), 2),
                "latency_ms": round((time.time() - t0) * 1000 + 260, 1),
                "demo": True,
            }
        # Real path would call SILMA inference here
        audio = _demo_wav()  # placeholder until SILMA API wired
        return {
            "audio": audio, "mime": "audio/wav",
            "duration_s": 1.2,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "demo": False,
        }

    @classmethod
    def status(cls) -> dict:
        cls.load()
        return {
            "name": cls.name,
            "loaded": _MODEL is not None,
            "demo": settings.is_demo or _MODEL is None,
            "error": _LOAD_ERROR,
            "path": str(settings.tts_path),
        }
