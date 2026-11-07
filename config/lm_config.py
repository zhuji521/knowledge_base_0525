import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    vl_model: str
    llm_model: str
    item_model: str
    llm_temperature: float

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_API_BASE") or "",
    api_key=os.getenv("OPENAI_API_KEY") or "",
    vl_model=os.getenv("VL_MODEL") or "",
    llm_model=os.getenv("LLM_DEFAULT_MODEL") or "qwen-flash",
    item_model=os.getenv("ITEM_MODEL") or os.getenv("LLM_DEFAULT_MODEL") or "qwen-flash",
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE") or "0.1"),
)