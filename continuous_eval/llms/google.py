import os
from typing import Dict, Optional

import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig
from google.generativeai.types.safety_types import (
    HarmCategory,
    LooseSafetySettingDict,
)

from .base import LLMInterface


class GoogleAI(LLMInterface):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        if api_key is None and os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError(
                "Please set the environment variable GOOGLE_API_KEY."
            )
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.safety_settings = [
            LooseSafetySettingDict(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold="block_none",
            ),
            LooseSafetySettingDict(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold="block_none",
            ),
            LooseSafetySettingDict(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold="block_none",
            ),
            LooseSafetySettingDict(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold="block_none",
            ),
        ]
        self.defaults = {
            "max_output_tokens": 2048,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 1.0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=prompt["system_prompt"],
            safety_settings=self.safety_settings,
            generation_config=GenerationConfig(**kwargs),
        )
        response = model.generate_content(prompt["user_prompt"])
        return response.text
