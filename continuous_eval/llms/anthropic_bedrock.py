import json

import boto3

from .base import LLMInterface, LLMInterfaceFactory


class AnthropicBedrock(LLMInterface):
    def __init__(
        self,
        model: str,
        anthropic_version: str,
        region_name: str,
        service_name: str = "bedrock-runtime",
        **kwargs,
    ):
        self.model = model
        self.anthropic_version = anthropic_version
        self.client = boto3.client(service_name, region_name=region_name)
        self.defaults = {
            "anthropic_version": anthropic_version,
            "max_tokens": 1024,
            "temperature": 1.0,
        }
        self.defaults.update(kwargs)

    def run(self, prompt: dict) -> str:
        body = self.defaults.copy()
        body["system"] = prompt["system_prompt"]
        body["messages"] = [{"role": "user", "content": prompt["user_prompt"]}]
        response = self.client.invoke_model(
            modelId=self.model, body=json.dumps(body)
        )
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]


class AnthropicBedrockFactory(LLMInterfaceFactory):
    def __init__(
        self,
        model: str,
        anthropic_version: str,
        region_name: str,
        service_name: str = "bedrock-runtime",
        **kwargs,
    ):
        self.model = model
        self.anthropic_version = anthropic_version
        self.region_name = region_name
        self.service_name = service_name
        self.defaults = {
            "max_tokens": 1024,
            "temperature": 1.0,
        }
        self.defaults.update(kwargs)

    def __call__(self, model, **kwargs):
        all_kwargs = {**self.defaults, **kwargs}
        return AnthropicBedrock(
            model=self.model,
            anthropic_version=self.anthropic_version,
            region_name=self.region_name,
            service_name=self.service_name,
            **all_kwargs,
        )
