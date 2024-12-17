import os
from typing import Dict, Optional

from .base import LLMInterface

try:
    from cohere import ClientV2, SystemChatMessageV2, UserChatMessageV2

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class Cohere(LLMInterface):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        if not COHERE_AVAILABLE:
            raise ValueError("Cohere is not available. Please install it.")
        if api_key is None and os.getenv("COHERE_API_KEY") is None:
            raise ValueError(
                "Please set the environment variable COHERE_API_KEY. "
                "You can get one at https://cohere.ai/."
            )
        self.client = ClientV2(api_key=api_key or os.getenv("COHERE_API_KEY"))  # type: ignore
        self.model = model
        self.defaults = {
            "seed": 0,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 1.0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.chat(
            model=self.model,
            messages=[
                SystemChatMessageV2(content=prompt["system_prompt"]),  # type: ignore
                UserChatMessageV2(content=prompt["user_prompt"]),
            ],
            **kwargs,
        )
        return response.message.content[0].text  # type: ignore
