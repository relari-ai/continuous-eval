from .base import LLMFactory
from .openai import OpenAI

LLMFactory.register_provider("openai", OpenAI)
