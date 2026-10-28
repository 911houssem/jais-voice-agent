"""Jais-3-7B wrapper (text -> text). Supports vllm / transformers / openai backends."""
from __future__ import annotations
import time
from typing import Optional, Iterator
from ..config import settings

_MODEL = None
_TOKENIZER = None
_LOADED = False
_LOAD_ERROR: Optional[str] = None

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


class LLMModel:
    name = "Jais-3-7B"
    _demo_idx = 0

    @classmethod
    def load(cls) -> None:
        global _MODEL, _TOKENIZER, _LOADED, _LOAD_ERROR
        if _LOADED:
            return
        if settings.is_demo:
            _LOAD_ERROR = "demo mode"
            _LOADED = True
            return
        backend = settings.llm_backend.lower()
        try:
            if backend == "vllm":
                # Lazy: defer to openai-compatible HTTP client
                _LOADED = True
                return
            if backend == "openai":
                _LOADED = True
                return
            # transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            ref = settings.ref(settings.llm_model_path)
            tok = settings.hf_token or None
            _TOKENIZER = AutoTokenizer.from_pretrained(ref, trust_remote_code=True, token=tok)
            load_kwargs = dict(trust_remote_code=True, torch_dtype="auto",
                               device_map=settings.device, token=tok)
            if settings.llm_load_in_8bit:
                load_kwargs.update(load_in_8bit=True)
            _MODEL = AutoModelForCausalLM.from_pretrained(ref, **load_kwargs)
            _LOADED = True
        except Exception as e:
            _LOAD_ERROR = f"{type(e).__name__}: {e}"
            _LOADED = True

    @classmethod
    def generate(cls, user_text: str, history: list | None = None) -> dict:
        cls.load()
        t0 = time.time()
        history = history or []
        if settings.is_demo or (_MODEL is None and settings.llm_backend not in ("vllm", "openai")):
            reply = DEMO_REPLIES[cls._demo_idx % len(DEMO_REPLIES)]
            cls._demo_idx += 1
            return {"text": reply, "latency_ms": round((time.time() - t0) * 1000 + 220, 1), "demo": True}

        backend = settings.llm_backend.lower()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_text}]
        try:
            if backend == "vllm" or backend == "openai":
                import httpx
                base = settings.vllm_base_url if backend == "vllm" else "https://api.openai.com/v1"
                key = settings.openai_api_key or "EMPTY"
                headers = {"Authorization": f"Bearer {key}"}
                model = settings.llm_path.name if backend == "vllm" else "gpt-4o-mini"
                r = httpx.post(f"{base}/chat/completions", headers=headers, json={
                    "model": model, "messages": messages, "max_tokens": 120, "temperature": 0.6,
                }, timeout=30)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
            else:
                import torch
                prompt = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = _TOKENIZER(prompt, return_tensors="pt").to(settings.device)
                with torch.no_grad():
                    out = _MODEL.generate(**inputs, max_new_tokens=settings.max_new_tokens,
                                          temperature=0.6, do_sample=True)
                text = _TOKENIZER.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"text": text.strip(), "latency_ms": round((time.time() - t0) * 1000, 1), "demo": False}
        except Exception as e:
            return {"text": f"[خطأ LLM: {e}]", "latency_ms": round((time.time() - t0) * 1000, 1), "demo": False, "error": str(e)}

    @classmethod
    def stream(cls, user_text: str, history: list | None = None) -> Iterator[str]:
        res = cls.generate(user_text, history)
        # Simulate streaming tokens for the UI
        for ch in res["text"]:
            yield ch

    @classmethod
    def status(cls) -> dict:
        cls.load()
        return {
            "name": cls.name,
            "loaded": _MODEL is not None or settings.llm_backend in ("vllm", "openai"),
            "backend": settings.llm_backend,
            "demo": settings.is_demo or _MODEL is None,
            "error": _LOAD_ERROR,
            "path": str(settings.llm_path),
        }
