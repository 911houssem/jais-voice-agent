FROM python:3.11-slim

WORKDIR /app

# libsndfile for soundfile (WAV I/O)
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; add the rest of the deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# HF Spaces must listen on 7860; run in real mode on CPU.
# Models (~30GB) are downloaded from the Hugging Face Hub at first use.
ENV MODE=real \
    DEVICE=cpu \
    PORT=7860 \
    ASR_MODEL_PATH=IFM/Jais-ASR \
    LLM_MODEL_PATH=inception42/Jais-3-7B \
    ASR_DTYPE=float16 \
    LLM_LOAD_IN_8BIT=true \
    TTS_BACKEND=edge \
    EDGE_VOICE=ar-SA-HamedNeural \
    MAX_NEW_TOKENS=90 \
    HF_HOME=/data/hf

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
