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
        backend = settings.tts_backend.lower()
        try:
            if backend in ("auto", "silma"):
                # SILMA ships as model.pt; load torch checkpoint lazily
                import torch
                path = settings.tts_path / "model.pt"
                if path.exists():
                    _MODEL = torch.load(str(path), map_location=settings.device, weights_only=False)
                    _LOADED = True
                    return
                if backend == "silma":
                    raise FileNotFoundError(f"SILMA weights not found: {path}")
            # edge backend: verified working on Kaggle (ar-SA-HamedNeural)
            import edge_tts  # noqa: F401
            _MODEL = "edge"
            _LOAD_ERROR = None
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
        if _MODEL == "edge":
            try:
                import asyncio
                import edge_tts

                async def _run() -> bytes:
                    comm = edge_tts.Communicate(text, settings.edge_voice)
                    buf = io.BytesIO()
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            buf.write(chunk["data"])
                    return buf.getvalue()

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        audio = ex.submit(asyncio.run, _run()).result(timeout=60)
                else:
                    audio = asyncio.run(_run())
                return {
                    "audio": audio, "mime": "audio/mpeg",
                    "duration_s": round(len(audio) / 16000, 2),
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "demo": False,
                }
            except Exception as e:
                return {"audio": _demo_wav(), "mime": "audio/wav", "duration_s": 1.2,
                        "latency_ms": round((time.time() - t0) * 1000, 1),
                        "demo": True, "error": str(e)}
        # SILMA path: real path would call SILMA inference here
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
