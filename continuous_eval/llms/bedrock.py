import json
import os
from typing import Dict

import boto3

from .base import LLMInterface


class Bedrock(LLMInterface):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.client = boto3.client(
            service_name=os.getenv("BEDROCK_SERVICE_NAME", "bedrock-runtime")
        )

    def run(
        self,
        prompt: Dict[str, str],
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        body = json.dumps(
            {
                "max_tokens": max_tokens,
                "top_k": 1,
                "stop_sequences": [],
                "temperature": temperature,
                "top_p": 0.999,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": prompt["system_prompt"]}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt["user_prompt"]}
                        ],
                    },
                ],
            }
        )
        response = self.client.invoke_model(body=body, modelId=self.model)
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")

        # self.client.model_kwargs["temperature"] = temperature
        # user_message = HumanMessage(content=prompt["user_prompt"])
        # sys_message = SystemMessage(content=prompt["system_prompt"])
        # response = self.client.invoke([sys_message, user_message])
        # return response.content
