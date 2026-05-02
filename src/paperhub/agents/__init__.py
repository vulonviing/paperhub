"""LLM clients and the per-paper agent."""

from .base import LLMClient, build_llm
from .paper_agent import PaperAgent
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_messages

__all__ = [
    "LLMClient",
    "PaperAgent",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "build_llm",
    "build_messages",
]
