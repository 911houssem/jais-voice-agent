"""FastAPI entrypoint: serves API + frontend."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.api.routes import router as api_router

FRONTEND = ROOT / "frontend"

app = FastAPI(title="Jais Voice Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def _warmup():
    """Load models in a background thread so the server binds its port instantly
    (HF Spaces kill containers that don't listen quickly) while the ~30GB of
    weights download lazily on first use."""
    import threading

    def _load_all():
        try:
            from backend.models.asr import ASRModel
            from backend.models.llm import LLMModel
            from backend.models.tts import TTSModel
            ASRModel.load()
            LLMModel.load()
            TTSModel.load()
        except Exception:
            pass  # /api/status surfaces per-model load errors

    threading.Thread(target=_load_all, daemon=True).start()


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND), html=True), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/{path:path}")
    def spa(path: str):
        candidate = FRONTEND / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True,
                app_dir=str(ROOT))
