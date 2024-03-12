import string

import pytest

from continuous_eval.llm_factory import LLMFactory
from continuous_eval.metrics.base import LLMBasedMetric


class DummyLLMMetric(LLMBasedMetric):
    def __init__(self, model):
        super().__init__(model)

    def calculate(self, **kwargs):
        prompt = {
            "system_prompt": (
                "You are just a dummy evaluator system for a question answering system."
                "Your task is to answer `yes` to everything."
            ),
            "user_prompt": ("Please tell me `yes`."),
        }
        return self._llm.run(prompt)


def test_llm_based_metric():
    models = [
        "gpt-3.5-turbo-1106",
        "claude-2.1",
        "gemini-pro",
        "cohere",
    ]

    for mdl in models:
        metric = DummyLLMMetric(LLMFactory(mdl))
        response = metric.calculate()
        assert (
            response is not None
            and response.lower()  # make sure the response is lowercased
            .strip()  # remove leading and trailing whitespaces
            .translate(str.maketrans("", "", string.punctuation))  # remove all punctuations
            == "yes"
        ), f"LLM based metric with model {mdl} failed."
