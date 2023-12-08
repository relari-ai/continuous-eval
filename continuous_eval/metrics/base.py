import os
from abc import ABC, abstractmethod

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()

EVAL_LLM = os.getenv("EVAL_LLM", "gpt-3.5-turbo-1106")


class Metric(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def requires(self):
        return


class LLMBasedMetric(Metric):
    """
    Base class for all LLM based metrics.
    """

    def __init__(self, model=EVAL_LLM):
        super().__init__()
        if model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-4-1106-preview"]:
            self.model = model
            self.client = OpenAI()
        elif model in ["claude-2.1", "claude-2.0", "claude-instant-1.2"]:
            self.client = Anthropic()
        else:
            raise ValueError(
                f"Model {model} is not supported. Please choose one of the following models: gpt-3.5-turbo-1106, gpt-4-1106-preview, claude-2.1, claude-2.0, claude-instant-1.2."
            )

    def _llm_response(self, prompt):
        """
        Send a prompt to the LLM and return the response.
        """
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["user_prompt"]},
                ],
                seed=0,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            content = response.choices[0].message.content
        elif isinstance(self.client, Anthropic):
            response = self.client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=1024,
                temperature=0.5,
                prompt=f"{prompt['system_prompt']}{HUMAN_PROMPT}{prompt['user_prompt']}{AI_PROMPT}",
            )
            content = response.completion
        else:
            raise ValueError(f"Unknown model client")

        return content
