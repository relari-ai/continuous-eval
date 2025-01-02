import os
from typing import Dict

from openai import OpenAI as _OpenAI

from .base import LLMInterface


class OpenAI(LLMInterface):
    def __init__(self, model: str, **kwargs):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the environment variable OPENAI_API_KEY. "
                "You can get one at https://beta.openai.com/account/api-keys."
            )
        self.client = _OpenAI()
        self.model = model
        self.defaults = {
            "seed": 0,
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 1.0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ],
            **kwargs,
        )
        return response.choices[0].message.content
