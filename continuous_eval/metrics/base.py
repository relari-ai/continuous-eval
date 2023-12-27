import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

try:
    import google.generativeai as google_genai

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


load_dotenv()

assert os.getenv("OPENAI_API_KEY") is not None, (
    "Please set the environment variable OPENAI_API_KEY. "
    "You can get one at https://beta.openai.com/account/api-keys."
)
if GOOGLE_GENAI_AVAILABLE:
    assert os.getenv("GEMINI_API_KEY") is not None, (
        "Please set the environment variable GEMINI_API_KEY. "
        "You can get one at https://ai.google.dev/tutorials/setup."
    )
    google_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if ANTHROPIC_AVAILABLE:
    assert os.getenv("ANTHROPIC_API_KEY") is not None, (
        "Please set the environment variable ANTHROPIC_API_KEY. " "You can get one at https://www.anthropic.com/signup."
    )


EVAL_LLM = os.getenv("EVAL_LLM", "gpt-3.5-turbo-1106")


class Metric(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError()

    def batch_calculate(self, dataset: List[Dict[str, Any]]):
        return [self.calculate(**datum) for datum in dataset]


class LLMBasedMetric(Metric):
    """
    Base class for all LLM based metrics.
    """

    def __init__(self, model=EVAL_LLM):
        super().__init__()
        self.model = model
        if model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-4-1106-preview"]:
            self.client = OpenAI()
        elif model in ["claude-2.1", "claude-2.0", "claude-instant-1.2"]:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic is not available. Please install the optional dependency `anthropic`.")
            self.client = Anthropic()
        elif model in ["gemini-pro"]:
            if not GOOGLE_GENAI_AVAILABLE:
                raise ImportError("Google GenAI is not available. Please install the optional dependency `gemini`.")
            self.client = google_genai.GenerativeModel(
                model_name=model,
            )
        else:
            raise ValueError(
                f"Model {model} is not supported. "
                "Please choose one of the following models: "
                "gpt-3.5-turbo-1106, gpt-3.5-turbo-16k, gpt-4-1106-preview, "
                "claude-2.1, claude-2.0, claude-instant-1.2, "
                "gemini-pro."
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
                temperature=0.0,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            content = response.choices[0].message.content
        elif isinstance(self.client, Anthropic):
            response = self.client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=1024,
                temperature=0.0,
                prompt=f"{prompt['system_prompt']}{HUMAN_PROMPT}{prompt['user_prompt']}{AI_PROMPT}",
            )
            content = response.completion
        elif isinstance(self.client, google_genai.GenerativeModel):
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 1024,
            }
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]
            response = self.client.generate_content(
                f"{prompt['system_prompt']}\n{prompt['user_prompt']}",
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            content = response.text
        else:
            raise ValueError(f"Unknown model client")

        return content
