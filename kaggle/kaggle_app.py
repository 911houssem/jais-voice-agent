"""Kaggle self-contained voice agent app (FastAPI + WebSocket + UI).

Studio flow: WebM/Opus from browser mic -> faster-whisper (ASR)
-> Jais-3-7B via transformers (device_map=auto) -> edge-tts -> WAV/MP3.

UI and message protocol are identical to the local backend, so
frontend/ works unchanged.
"""
from __future__ import annotations
import base64
import io
import json
import math
import os
import shutil
import struct
import sys
import threading
import time
import asyncio
import base64 as _b64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent            # voice-agent/
FRONTEND = ROOT / "frontend"

CFG_FILE = Path("/kaggle/working/.jais_config.json")
CONFIG = {}
for _c in ([Path(os.environ["JAIS_CONFIG"])] if os.environ.get("JAIS_CONFIG") else []) + [
    Path("/content/.jais_config.json"),      # Google Colab
    Path("/kaggle/working/.jais_config.json") # Kaggle
]:
    if _c.exists():
        CONFIG = json.loads(_c.read_text(encoding="utf-8"))
        break

PORT = int(CONFIG.get("PORT", 8000))
ASR_SIZE = CONFIG.get("ASR_SIZE", "small")
ASR_DEVICE = CONFIG.get("ASR_DEVICE", "cpu")
LLM_MODEL = CONFIG.get("LLM_MODEL", "inception42/Jais-3-7B")
TTS_VOICE = CONFIG.get("TTS_VOICE", "ar-SA-HamedNeural")

SYSTEM_PROMPT = (
    "أنت وكيل صوتي لخدمة العملاء باللغة العربية. "
    "أجب بوضوح وبشكل مختصر (1-3 جمل). "
    "إذا طلب العميل حجز موعد، اطلب: الاسم، التاريخ، الوقت. "
    "لا تختلق معلومات — إذا لم تعرف، قل أنك ستحوله لموظف."
)

DEMO_REPLIES = [
    "أهلاً بك! كيف يمكنني مساعدتك اليوم؟",
    "بالتأكيد، سأساعدك في حجز موعد. ما اسمك الكامل؟",
    "ممتاز، أي يوم يناسبك؟",
    "تم حجز موعدك. هل هناك أي شيء آخر؟",
    "شكراً لاتصالك. يوم سعيد!",
]

# ---------------------------------------------------------------------------
# ASR (faster-whisper, self-downloads on first use)
# ---------------------------------------------------------------------------
_asr = None
_asr_err = None

def _get_asr():
    global _asr, _asr_err
    if _asr is not None or _asr_err is not None:
        return _asr
    try:
        from faster_whisper import WhisperModel
        _asr = WhisperModel(ASR_SIZE, device=ASR_DEVICE, compute_type="int8")
    except Exception as e:  # noqa
        _asr_err = f"{type(e).__name__}: {e}"
    return _asr

def transcribe(audio_bytes: bytes) -> dict:
    t0 = time.time()
    model = _get_asr()
    if model is None:
        return {"text": "", "language": "arb", "latency_ms": round((time.time()-t0)*1000,1), "demo": True}
    try:
        bi = io.BytesIO(audio_bytes)
        segments, info = model.transcribe(bi, language=None, beam_size=2)
        text = " ".join(s.text for s in segments).strip()
        return {"text": text, "language": info.language or "arb",
                "latency_ms": round((time.time()-t0)*1000,1), "demo": False}
    except Exception as e:  # noqa
        return {"text": "", "language": "arb", "latency_ms": round((time.time()-t0)*1000,1),
                "demo": False, "error": str(e)}

# ---------------------------------------------------------------------------
# LLM (Jais via transformers, lazy)
# ---------------------------------------------------------------------------
_llm = None
_tok = None
_llm_err = None
_demo_idx = 0

def _get_llm():
    global _llm, _tok, _llm_err
    if _llm is not None or _llm_err is not None:
        return _llm
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        _tok = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        _llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, trust_remote_code=True, torch_dtype=dtype,
            device_map=device_map, low_cpu_mem_usage=True,
        )
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
    except Exception as e:  # noqa
        _llm_err = f"{type(e).__name__}: {e}"
    return _llm

def llm_generate(user_text: str, history: list) -> dict:
    global _demo_idx
    t0 = time.time()
    model = _get_llm()
    if model is None:
        r = DEMO_REPLIES[_demo_idx % len(DEMO_REPLIES)]
        _demo_idx += 1
        return {"text": r, "latency_ms": round((time.time()-t0)*1000+200,1), "demo": True,
                "error": _llm_err}
    try:
        import torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-12:] + [{"role": "user", "content": user_text}]
        prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=True,
                                 temperature=0.6, top_p=0.9,
                                 pad_token_id=_tok.pad_token_id)
        text = _tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {"text": text, "latency_ms": round((time.time()-t0)*1000,1), "demo": False}
    except Exception as e:  # noqa
        return {"text": f"[خطأ LLM: {e}]", "latency_ms": round((time.time()-t0)*1000,1),
                "demo": False, "error": str(e)}

# ---------------------------------------------------------------------------
# TTS (edge-tts; falls back to a demo sine tone offline)
# ---------------------------------------------------------------------------
_tts_err = None

def _demo_wav(duration_s: float = 1.2, freq: float = 220.0) -> bytes:
    sr = 16000
    n = int(sr * duration_s)
    buf = io.BytesIO()
    data_size = n * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for i in range(n):
        v = int(32767 * 0.25 * math.sin(2*math.pi*freq*i/sr) * (1 - i/n))
        buf.write(struct.pack("<h", v))
    return buf.getvalue()

def synthesize(text: str) -> dict:
    t0 = time.time()
    try:
        import edge_tts
        import asyncio

        def _run():
            async def _go():
                c = edge_tts.Communicate(text, TTS_VOICE, rate="-5%")
                buf = io.BytesIO()
                async for chunk in c.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                return buf.getvalue()
            return asyncio.run(_go())

        audio = _run()
        return {"audio": audio, "mime": "audio/mp3",
                "duration_s": round(len(audio)/24000.0, 2),
                "latency_ms": round((time.time()-t0)*1000,1), "demo": False}
    except Exception as e:  # noqa
        wav = _demo_wav(min(2.5, 0.4 + 0.08*len(text)))
        return {"audio": wav, "mime": "audio/wav",
                "duration_s": round(len(wav)/32000,2),
                "latency_ms": round((time.time()-t0)*1000,1),
                "demo": True, "error": f"{type(e).__name__}: {e}"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Jais Voice (Kaggle)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class History:
    def __init__(self):
        self.items: list = []
    def push(self, role: str, content: str):
        self.items.append({"role": role, "content": content})
        self.items = self.items[-12:]
    def clear(self):
        self.items.clear()

_http_hist = History()

class TurnReq(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/status")
def status():
    return {
        "mode": "kaggle",
        "models": {
            "asr": {"name": f"faster-whisper:{ASR_SIZE}", "loaded": _asr is not None, "demo": _asr is None, "error": _asr_err},
            "llm": {"name": LLM_MODEL, "loaded": _llm is not None, "backend": "transformers", "demo": _llm is None, "error": _llm_err},
            "tts": {"name": f"edge-tts:{TTS_VOICE}", "loaded": True, "demo": False, "error": ""},
        },
        "max_concurrent_calls": 2,
    }

@app.post("/api/chat")
def chat(req: TurnReq):
    return llm_generate(req.text, _http_hist.items)

@app.post("/api/tts")
def tts(req: TurnReq):
    r = synthesize(req.text)
    return {"audio_b64": base64.b64encode(r["audio"]).decode(), "mime": r["mime"],
            "duration_s": r["duration_s"], "latency_ms": r["latency_ms"], "demo": r.get("demo", True)}

@app.post("/api/turn")
def turn(req: TurnReq):
    t0 = time.time()
    llm_r = llm_generate(req.text, _http_hist.items)
    reply = llm_r["text"]
    _http_hist.push("user", req.text)
    _http_hist.push("assistant", reply)
    tts_r = synthesize(reply)
    return {
        "user_text": req.text,
        "reply_text": reply,
        "audio_b64": base64.b64encode(tts_r["audio"]).decode(),
        "mime": tts_r["mime"],
        "duration_s": tts_r["duration_s"],
        "timings": {"asr_ms": 0, "llm_ms": llm_r["latency_ms"],
                    "tts_ms": tts_r["latency_ms"],
                    "total_ms": round((time.time()-t0)*1000,1)},
        "demo": tts_r.get("demo", True),
    }

@app.post("/api/turn/reset")
def turn_reset():
    _http_hist.clear()
    return {"ok": True}

@app.post("/api/asr")
async def asr_upload(file: UploadFile = File(...)):
    data = await file.read()
    return transcribe(data)

# ---- WebSocket voice session (same protocol as local backend) ----
class VoiceSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.hist = History()

    async def send(self, payload: dict):
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def run(self):
        await self.ws.accept()
        await self.send({"type": "ready", "mode": "kaggle", "msg": "جاهز أسمعك... تكلم الآن"})
        try:
            while True:
                data = await self.ws.receive()
                if "text" in data and data["text"] is not None:
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "reset":
                            self.hist.clear()
                            await self.send({"type": "reset_ok"})
                            continue
                        if msg.get("type") == "ping":
                            await self.send({"type": "pong"})
                            continue
                    except Exception:
                        pass
                    continue
                if "bytes" in data and data["bytes"] is not None:
                    await self.process_turn(data["bytes"])
                    continue
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa
            try:
                await self.send({"type": "error", "error": str(e)})
            except Exception:
                pass

    async def process_turn(self, audio_bytes: bytes):
        loop = asyncio.get_event_loop()
        t0 = time.time()
        await self.send({"type": "state", "state": "listening", "msg": "أفهمك..."})
        asr_r = await loop.run_in_executor(None, transcribe, audio_bytes)
        user_text = asr_r.get("text", "")
        asr_ms = asr_r.get("latency_ms", 0)
        await self.send({"type": "transcript", "role": "user", "text": user_text,
                         "latency_ms": asr_ms, "demo": asr_r.get("demo", True)})
        if not user_text.strip():
            await self.send({"type": "state", "state": "idle", "msg": "ما سمعت شي واضح، جرب مرة ثانية."})
            return
        await self.send({"type": "state", "state": "thinking", "msg": "أفكر..."})
        llm_r = await loop.run_in_executor(None, lambda: llm_generate(user_text, self.hist.items))
        reply = llm_r["text"]
        self.hist.push("user", user_text)
        self.hist.push("assistant", reply)
        await self.send({"type": "transcript", "role": "agent", "text": reply,
                         "latency_ms": llm_r["latency_ms"], "demo": llm_r.get("demo", True)})
        await self.send({"type": "state", "state": "speaking", "msg": "أرد الآن..."})
        tts_r = await loop.run_in_executor(None, lambda: synthesize(reply))
        total_ms = round((time.time()-t0)*1000, 1)
        await self.send({
            "type": "answer",
            "reply_text": reply,
            "audio_b64": base64.b64encode(tts_r["audio"]).decode(),
            "mime": tts_r["mime"],
            "duration_s": tts_r.get("duration_s", 0),
            "timings": {"asr_ms": asr_ms, "llm_ms": llm_r["latency_ms"],
                        "tts_ms": tts_r["latency_ms"], "total_ms": total_ms},
            "demo": tts_r.get("demo", True),
        })
        await self.send({"type": "state", "state": "idle", "msg": "جاهز أسمعك..."})

@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await VoiceSession(ws).run()

# ---- Serve frontend ----
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND), html=True), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/{path:path}")
    def spa(path: str):
        c = FRONTEND / path
        if c.is_file():
            return FileResponse(str(c))
        return FileResponse(str(FRONTEND / "index.html"))

# ---------------------------------------------------------------------------
# Boot helper (uvicorn in a daemon thread)
# ---------------------------------------------------------------------------
_server = None

def start_app(port: int = PORT):
    global _server
    if _server is not None:
        print(f"App already running on port {PORT}")
        return
    os.environ.setdefault("KAGGLE_VOICE_PORT", str(port))
    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    _server = uvicorn.Server(cfg)
    th = threading.Thread(target=_server.run, daemon=True)
    th.start()
    deadline = time.time() + 30
    while not getattr(_server, "started", False) and time.time() < deadline:
        time.sleep(0.2)
    print(f"[OK] Jais Voice running on http://127.0.0.1:{port}")
