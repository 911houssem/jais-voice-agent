"""Application configuration loaded from environment / .env file."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent


class Settings(BaseSettings):
    mode: str = Field(default="demo", alias="MODE")
    asr_model_path: str = Field(default="../IFMJais-ASR", alias="ASR_MODEL_PATH")
    llm_model_path: str = Field(default="../inception42Jais-3-7B", alias="LLM_MODEL_PATH")
    tts_model_path: str = Field(default="../silma tts", alias="TTS_MODEL_PATH")
    llm_backend: str = Field(default="transformers", alias="LLM_BACKEND")
    vllm_base_url: str = Field(default="http://localhost:8001/v1", alias="VLLM_BASE_URL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    device: str = Field(default="cuda", alias="DEVICE")
    max_concurrent_calls: int = Field(default=45, alias="MAX_CONCURRENT_CALLS")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    # HuggingFace Space / cloud
    hf_token: str = Field(default="", alias="HF_TOKEN")
    llm_load_in_8bit: bool = Field(default=False, alias="LLM_LOAD_IN_8BIT")
    asr_torch_dtype: str = Field(default="auto", alias="ASR_DTYPE")
    tts_backend: str = Field(default="auto", alias="TTS_BACKEND")  # auto|silma|edge|demo
    edge_voice: str = Field(default="ar-SA-HamedNeural", alias="EDGE_VOICE")
    max_new_tokens: int = Field(default=120, alias="MAX_NEW_TOKENS")

    class Config:
        env_file = str(ROOT / ".env")
        env_file_encoding = "utf-8"
        populate_by_name = True

    def resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return path

    def ref(self, p: str) -> str:
        """Local path if it exists on disk, otherwise treat the value as a HF repo id."""
        path = self.resolve(p)
        return str(path) if path.exists() else p

    @property
    def asr_path(self) -> Path:
        return self.resolve(self.asr_model_path)

    @property
    def llm_path(self) -> Path:
        return self.resolve(self.llm_model_path)

    @property
    def tts_path(self) -> Path:
        return self.resolve(self.tts_model_path)

    @property
    def is_demo(self) -> bool:
        return self.mode.lower() != "real"


settings = Settings()
