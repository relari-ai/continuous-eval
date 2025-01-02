import os

from dotenv import load_dotenv

from continuous_eval.llms.anthropic import Anthropic
from continuous_eval.llms.anthropic_bedrock import AnthropicBedrockFactory
from continuous_eval.llms.azure import AzureAIFactory
from continuous_eval.llms.azure_openai import AzureOpenAIFactory
from continuous_eval.llms.base import _LLMFactory
from continuous_eval.llms.cohere import Cohere
from continuous_eval.llms.google import GoogleAI
from continuous_eval.llms.openai import OpenAI

load_dotenv()

_PROMPT = {
    "system_prompt": "You are a helpful assistant.",
    "user_prompt": "What is the capital of France?",
}


def test_openai():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider("openai", OpenAI)
    llm = _llm_factory.get("openai:gpt-4o-mini")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_azure():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider(
        "azure",
        model="azure_test_model",
        provider_class=AzureAIFactory(
            endpoint=os.getenv("AZURE_ENDPOINT"),
            credential=os.getenv("AZURE_CREDENTIAL"),
        ),
    )
    llm = _llm_factory.get("azure:azure_test_model")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_azure_openai():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider(
        "azure_openai",
        model="gpt-4o-mini",
        provider_class=AzureOpenAIFactory(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        ),
    )
    llm = _llm_factory.get("azure_openai:gpt-4o-mini")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_anthropic_bedrock():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider(
        "anthropic_bedrock",
        model="claude-3-5-haiku",
        provider_class=AnthropicBedrockFactory(
            model="anthropic.claude-3-5-haiku-20241022-v1:0",
            anthropic_version="bedrock-2023-05-31",
            region_name="us-west-2",
        ),
    )
    llm = _llm_factory.get("anthropic_bedrock:claude-3-5-haiku")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_anthropic():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider("anthropic", Anthropic)
    llm = _llm_factory.get("anthropic:claude-3-5-sonnet-20241022")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_cohere():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider("cohere", Cohere)
    llm = _llm_factory.get("cohere:command-r-plus")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0


def test_google():
    _llm_factory = _LLMFactory()
    _llm_factory.register_provider("google", GoogleAI)
    llm = _llm_factory.get("google:gemini-1.5-flash-002")
    res = llm.run(_PROMPT)
    assert res is not None and isinstance(res, str) and len(res) > 0
