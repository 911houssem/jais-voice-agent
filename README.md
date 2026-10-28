# Jais Voice — وكيل صوتي عربي (ASR → LLM → TTS)

منصة متكاملة لـ **أتمتة مكالمات العملاء باللغة العربية**، تدمج three نماذج مفتوحة (Apache-2.0) في خط أنبابة واحد:

```
📞 مكالمة  →  🎙️ Jais-ASR  →  🧠 Jais-3-7B  →  🔊 SILMA TTS  →  📞 رد صوتي
```

- **Jais-ASR** — تحويل كلام (لهجات عربية) → نص
- **Jais-3-7B** — توليد الرد (عربي/إنجليزي، 512K سياق)
- **SILMA TTS** — تحويل الرد → صوت (استنساخ صوت < 8 ثوانٍ)

## لماذا this architecture

- **أوزان مفتوحة** → لا رسوم per-token لجهات خارجية
- **استضافة داخلية ممكنة** → سيادة بيانات (PDPL / جهات حكومية)
- **كلفة ثابتة** → هامش ~80% بعد نقطة التعادل (~6–7 مكالمات متزامنة)

## هيكل المشروع

```
voice-agent/
├── backend/          # FastAPI — API + WebSocket + pipeline
│   ├── main.py
│   ├── config.py
│   ├── models/       # wrappers: asr.py, llm.py, tts.py
│   ├── pipeline/     # orchestrator.py
│   └── api/          # routes, schemas, websocket
├── frontend/         # واجهة احترافية (RTL Arabic, dark)
├── docker/           # Dockerfile + compose
├── scripts/          # run_dev.bat / .sh
└── .env.example
```

## التشغيل (وضع Demo — بدون GPU)

```bash
cd backend
pip install -r requirements.txt
python main.py
# افتح http://localhost:8000
```

الواجهة تعمل في **وضع Demo** (mock streaming) لعرض التصميم والتدفق دون GPU.
لتفعيل النماذج الحقيقية، عيّن `MODE=real` في `.env` وثبّت `transformers`, `vllm`, `silma-tts`.

## التشغيل (وضع Real — مع GPU)

```bash
# 1. ثبّت الاعتماديات الإضافية
pip install transformers accelerate vllm soundfile

# 2. انسخ .env.example → .env وحدّث مسارات النماذج
cp .env.example .env

# 3. شغّل
python main.py
```

## متغيرات البيئة (.env)

```
MODE=demo                      # demo | real
ASR_MODEL_PATH=../IFMJais-ASR
LLM_MODEL_PATH=../inception42Jais-3-7B
TTS_MODEL_PATH=../silma tts
LLM_BACKEND=vllm               # vllm | transformers | openai
VLLM_BASE_URL=http://localhost:8001/v1
DEVICE=cuda                    # cuda | cpu
MAX_CONCURRENT_CALLS=45
```

## الترخيص

النماذج (Jais-3-7B, Jais-ASR, SILMA TTS) مرخصة **Apache-2.0** — استخدام تجاري مسموح.
كود هذا المشروع: MIT.

## خارطة الطريق

- [x] خط أنبابة ASR → LLM → TTS (demo + real hooks)
- [x] واجهة احترافية (RTL, dark, dashboard)
- [ ] تكامل SIP/Telephony (FreeSWITCH)
- [ ] مصادقة + multi-tenant
- [ ] فوترة + باقات
- [ ] on-prem installer
