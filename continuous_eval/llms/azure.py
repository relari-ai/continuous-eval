import os
from typing import Dict, Optional

from .base import LLMInterface, LLMInterfaceFactory

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class AzureAI(LLMInterface):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[str] = None,
        **kwargs,
    ):
        if not AZURE_AVAILABLE:
            raise ValueError("Azure OpenAI is not available")
        if os.getenv("AZURE_ENDPOINT") is None and endpoint is None:
            raise ValueError(
                "Please set endpoint or the environment variable AZURE_ENDPOINT"
            )
        if os.getenv("AZURE_CREDENTIAL") is None and credential is None:
            raise ValueError(
                "Please set credential or the environment variable AZURE_CREDENTIAL"
            )
        _endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
        _credential = AzureKeyCredential(
            credential or os.getenv("AZURE_CREDENTIAL")
        )  # type: ignore
        self.client = ChatCompletionsClient(
            endpoint=_endpoint,  # type: ignore
            credential=_credential,
        )
        self.defaults = {
            "temperature": 1.0,
            "max_tokens": 2048,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.complete(
            messages=[
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class AzureAIFactory(LLMInterfaceFactory):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[str] = None,
        **kwargs,
    ):
        self.endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
        self.credential = credential or os.getenv("AZURE_CREDENTIAL")
        self.extra_kwargs = kwargs

    def __call__(self, model, **kwargs):
        all_kwargs = {**self.extra_kwargs, **kwargs}
        return AzureAI(
            endpoint=self.endpoint,
            credential=self.credential,
            **all_kwargs,
        )
