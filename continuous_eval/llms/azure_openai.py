import os
from typing import Dict, Optional

from .base import LLMInterface, LLMInterfaceFactory

try:
    from openai import AzureOpenAI as _AzureOpenAI

    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False


class AzureOpenAI(LLMInterface):
    """
    Azure OpenAI LLM provider.

    Example:
    ```
    llm = AzureOpenAI(
        api_key="1234567890123",
        api_version="2024-08-01-preview",
        endpoint="https://example-endpoint.openai.azure.com/",
        deployment="gpt-4o-mini",
    )
    print(llm.run({"system_prompt": "You are a helpful assistant.","user_prompt": "What is the capital of France?"}))
    ```

    To register a new provider, use the following:
    ```
    LLMFactory.register_provider(
        "azure_openai",
        model="gpt-4o-mini",
        provider_class=AzureOpenAIFactory(
            api_key="1234567890123",
            api_version="2024-08-01-preview",
            endpoint="https://example-endpoint.openai.azure.com/",
            deployment="gpt-4o-mini",
        ),
    )
    llm = LLMFactory.get("azure_openai:gpt-4o-mini")
    print(llm.run({"system_prompt": "You are a helpful assistant.", "user_prompt": "What is the capital of France?"}))
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        **kwargs,
    ):
        if not AZURE_OPENAI_AVAILABLE:
            raise ValueError("Azure OpenAI is not available")
        if os.getenv("AZURE_OPENAI_API_KEY") is None and api_key is None:
            raise ValueError(
                "Please set the environment variable AZURE_OPENAI_API_KEY. "
                "You can get one at https://portal.azure.com."
            )
        if (
            os.getenv("AZURE_OPENAI_API_VERSION") is None
            and api_version is None
        ):
            raise ValueError(
                "Please set the environment variable AZURE_OPENAI_API_VERSION. "
                "You can get one at https://portal.azure.com."
            )
        if os.getenv("AZURE_ENDPOINT") is None and endpoint is None:
            raise ValueError(
                "Please set the environment variable AZURE_ENDPOINT. "
                "You can get one at https://portal.azure.com."
            )
        if os.getenv("AZURE_DEPLOYMENT") is None and deployment is None:
            raise ValueError(
                "Please set the environment variable AZURE_DEPLOYMENT. "
                "You can get one at https://portal.azure.com."
            )
        self.client = _AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=endpoint or os.getenv("AZURE_ENDPOINT"),  # type: ignore
            azure_deployment=deployment or os.getenv("AZURE_DEPLOYMENT"),
        )
        self.defaults = {
            "seed": 0,
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(
            model="<ignored>",
            messages=[
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class AzureOpenAIFactory(LLMInterfaceFactory):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
        self.deployment = deployment or os.getenv("AZURE_DEPLOYMENT")
        self.extra_kwargs = kwargs

    def __call__(self, model, **kwargs):
        all_kwargs = {**self.extra_kwargs, **kwargs}
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            endpoint=self.endpoint,
            deployment=self.deployment,
            **all_kwargs,
        )
