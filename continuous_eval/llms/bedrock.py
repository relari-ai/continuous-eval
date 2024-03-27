from typing import Dict

from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage, SystemMessage

from continuous_eval.llm_factory import LLMInterface


class Bedrock(LLMInterface):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.client = BedrockChat(model_id=model, model_kwargs={"temperature": 0.0})

    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        self.client.model_kwargs["temperature"] = temperature
        user_message = HumanMessage(content=prompt["user_prompt"])
        sys_message = SystemMessage(content=prompt["system_prompt"])
        response = self.client.invoke([sys_message, user_message])
        return response.content
