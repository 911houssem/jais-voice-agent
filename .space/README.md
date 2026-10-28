---
title: Jais Voice Agent
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - audio
  - arabic
  - voice-agent
  - jais
  - asr
  - tts
  - fastapi
---

# 🎙️ وكيل Jais الصوتي — Jais Voice Agent

وكيل صوتي عربي متكامل: **تفهم كلامك (ASR) ← تفكّر وتجيب (LLM) ← تتكلّم (TTS)**.

| المرحلة | النموذج | المعيار |
|---|---|---|
| التعرّف على الكلام | [IFM/Jais-ASR](https://huggingface.co/IFM/Jais-ASR) | Apache-2.0 |
| توليد اللغة | [inception42/Jais-3-7B](https://huggingface.co/inception42/Jais-3-7B) | Apache-2.0 |
| تحويل النص إلى كلام | edge-tts `ar-SA-HamedNeural` (بديل SILMA) | — |

> 🕐 **أول تشغيل:** تُنزَّل النماذج (~30GB) من Hugging Face تلقائيًا — قد يستغرق الأمر عدة دقائق قبل أن تعمل المكالمات بالوضع الحقيقي. تقدّم التحميل متاح في `/api/status`.

## الاستخدام

- الواجهة العربية الكاملة على الصفحة الرئيسية `/`
- `GET /api/status` — حالة النماذج الثلاثة
- `POST /api/turn` — دورة كاملة: `{ "text": "..." }` ← رد نصي + صوت (base64)

## التشغيل محليًا

```bash
cd voice-agent
pip install -r backend/requirements.txt
MODE=real python -m uvicorn backend.main:app --port 8000
```

## الرخصة والتجارة

النماذج الثلاثة برخصة **Apache-2.0** → استخدام تجاري مسموح. اقتراح تسعير: **$0.12/دقيقة**.

المستودع الكامل (كود + نوتة Kaggle + Docker): [github.com/911houssem/jais-voice-agent](https://github.com/911houssem/jais-voice-agent)
