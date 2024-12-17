import os
from typing import Dict, Optional

from .base import LLMInterface

try:
    from anthropic import Anthropic as _Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class Anthropic(LLMInterface):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic is not available")
        if api_key is None and os.getenv("ANTHROPIC_API_KEY") is None:
            raise ValueError(
                "Please set the environment variable ANTHROPIC_API_KEY. "
                "You can get one at https://www.anthropic.com/signup."
            )
        self.client = _Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.defaults = {
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 1.0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.messages.create(
            model=self.model,
            system=prompt["system_prompt"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt["user_prompt"]}
                    ],
                }
            ],
            **kwargs,
        )
        return response.content[0].text
