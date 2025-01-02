import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Union


class LLMInterface(ABC):
    @abstractmethod
    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        pass


class LLMInterfaceFactory:
    @abstractmethod
    def __call__(self, model: str, **kwargs) -> LLMInterface:
        pass


class _LLMFactory:
    def __init__(self):
        self.providers = defaultdict(dict)

    def register_provider(
        self,
        provider: str,
        provider_class: Union[type[LLMInterface], LLMInterfaceFactory],
        model: Optional[str] = "*",
    ):
        self.providers[provider][model] = provider_class

    @staticmethod
    def default() -> str:
        return os.getenv("DEFAULT_EVAL_MODEL", "openai:gpt-4o-mini")

    def get(self, model: Optional[str] = None, **kwargs) -> LLMInterface:
        if model is None:
            model = self.default()
        _split = model.split(":")
        provider = _split[0]
        model = model[len(provider) + 1 :]
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        if model in self.providers[provider]:
            return self.providers[provider][model](model, **kwargs)
        elif "*" in self.providers[provider]:
            return self.providers[provider]["*"](model, **kwargs)
        raise ValueError(f"Model {model} not found for provider {provider}")


LLMFactory = _LLMFactory()
