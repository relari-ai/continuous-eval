import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential  # for exponential backoff

load_dotenv()


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
try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_community.chat_models import AzureChatOpenAI

    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
try:
    from cohere import Client as CohereClient

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class LLMInterface(ABC):
    @abstractmethod
    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        pass


class LLMFactory(LLMInterface):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if model.startswith("gpt"):
            assert os.getenv("OPENAI_API_KEY") is not None, (
                "Please set the environment variable OPENAI_API_KEY. "
                "You can get one at https://beta.openai.com/account/api-keys."
            )
            self.client = OpenAI()
        elif model.startswith("claude"):
            assert ANTHROPIC_AVAILABLE, "Anthropic is not available. Please install it."
            assert os.getenv("ANTHROPIC_API_KEY") is not None, (
                "Please set the environment variable ANTHROPIC_API_KEY. "
                "You can get one at https://www.anthropic.com/signup."
            )
            self.client = Anthropic()
        elif model.startswith("gemini"):
            assert GOOGLE_GENAI_AVAILABLE, "Google GenAI is not available. Please install it."
            assert os.getenv("GEMINI_API_KEY") is not None, (
                "Please set the environment variable GEMINI_API_KEY. "
                "You can get one at https://ai.google.dev/tutorials/setup."
            )
            google_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = google_genai.GenerativeModel(model_name=model)
        elif model.startswith("cohere"):
            assert COHERE_AVAILABLE, "Cohere is not available. Please install it."
            assert os.getenv("COHERE_API_KEY") is not None, (
                "Please set the environment variable COHERE_API_KEY. " "You can get one at https://cohere.ai/."
            )
            self.client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))  # type: ignore
        elif model.startswith("azure"):
            assert AZURE_OPENAI_AVAILABLE, "Azure OpenAI is not available. Please install it."
            assert os.getenv("AZURE_OPENAI_API_KEY") is not None, (
                "Please set the environment variable AZURE_OPENAI_API_KEY. "
                "You can get one at https://portal.azure.com."
            )
            assert os.getenv("AZURE_OPENAI_API_VERSION") is not None, (
                "Please set the environment variable AZURE_OPENAI_API_VERSION. "
                "You can get one at https://portal.azure.com."
            )
            assert os.getenv("AZURE_ENDPOINT") is not None, (
                "Please set the environment variable AZURE_ENDPOINT. " "You can get one at https://portal.azure.com."
            )
            assert os.getenv("AZURE_DEPLOYMENT") is not None, (
                "Please set the environment variable AZURE_DEPLOYMENT. " "You can get one at https://portal.azure.com."
            )
            self.client = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # type: ignore
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # type: ignore
            )
        elif model.startswith("bedrock:"):
            from continuous_eval.llms.bedrock import Bedrock

            model_name = model.split(":")
            self.client = Bedrock(model_name[1])
            self.model = model_name[1]
        elif model.startswith("vllm_"):
            # use with VLLM_BASE_URL=http://localhost:8000/v1/ EVAL_LLM=vllm_Qwen/Qwen1.5-14B-Chat-GPTQ-Int8 python ...
            model_name = model.split("_")[1]
            self.model = model_name
            base_url = os.getenv("VLLM_BASE_URL")
            assert os.getenv("VLLM_BASE_URL") is not None, "Please set the environment variable VLLM_BASE_URL. "
            self.client = OpenAI(base_url=base_url, api_key="unused")
        else:
            raise ValueError(
                f"Model {model} is not supported. "
                "Please choose from one of the following LLM providers: "
                "OpenAI gpt models (e.g. gpt-4-turbo-preview, gpt-3.5-turbo-0125), Anthropic claude models (e.g. claude-2.1, claude-instant-1.2), Google Gemini models (e.g. gemini-pro), Azure OpenAI deployment (azure)"
            )

    @retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(15))
    def _llm_response(self, prompt, temperature):
        """
        Send a prompt to the LLM and return the response.
        """
        if isinstance(self.client, OpenAI):
            # Leverage JSON mode in OpenAI API. Make sure the system prompt contains "Output JSON".
            if "Output JSON" in prompt["system_prompt"] and self.client.api_key != "unused":
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt["system_prompt"]},
                        {"role": "user", "content": prompt["user_prompt"]},
                    ],
                    seed=0,
                    temperature=temperature,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt["system_prompt"]},
                        {"role": "user", "content": prompt["user_prompt"]},
                    ],
                    seed=0,
                    temperature=temperature,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            content = response.choices[0].message.content
        elif ANTHROPIC_AVAILABLE and isinstance(self.client, Anthropic):
            response = self.client.completions.create(  # type: ignore
                model="claude-2.1",
                max_tokens_to_sample=1024,
                temperature=temperature,
                prompt=f"{prompt['system_prompt']}{HUMAN_PROMPT}{prompt['user_prompt']}{AI_PROMPT}",
            )
            content = response.completion
        elif COHERE_AVAILABLE and isinstance(self.client, CohereClient):
            prompt = f"{prompt['system_prompt']}\n{prompt['user_prompt']}"
            response = self.client.generate(model="command", prompt=prompt, temperature=temperature, max_tokens=1024)  # type: ignore
            try:
                content = response.generations[0].text
            except:
                content = ""
                warnings.warn(f"Failed to generate content from Cohere")
        elif GOOGLE_GENAI_AVAILABLE and isinstance(self.client, google_genai.GenerativeModel):
            generation_config = {
                "temperature": temperature,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 1024,
            }
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ]
            response = self.client.generate_content(
                f"{prompt['system_prompt']}\n{prompt['user_prompt']}",
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            content = response.text
        elif AZURE_OPENAI_AVAILABLE and isinstance(self.client, AzureChatOpenAI):
            response = self.client.invoke(
                input=[
                    SystemMessage(content=prompt["system_prompt"]),
                    HumanMessage(content=prompt["user_prompt"]),
                ],
                temperature=temperature,
                max_tokens=1024,
                top_p=1,
            )
            content = response.dict()["content"]
        elif isinstance(self.client, LLMInterface):
            content = self.client.run(prompt=prompt, temperature=temperature)
        else:
            raise ValueError(f"Unknown model client")

        return content

    def run(self, prompt, temperature=0):
        """
        Run the LLM and return the response.
        Default temperature: 0
        """
        content = self._llm_response(prompt=prompt, temperature=temperature)
        return content


DefaultLLM = lambda: LLMFactory(model=os.getenv("EVAL_LLM", "gpt-3.5-turbo-0125"))
