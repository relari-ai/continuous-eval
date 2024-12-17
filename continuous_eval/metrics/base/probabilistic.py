import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Generic, TypeVar

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from continuous_eval.metrics.base import Field as MetricField
from continuous_eval.metrics.base.prompt import MetricPrompt

DEFAULT_MODEL = os.getenv(
    "DEFAULT_PROBABILISTIC_METRIC_MODEL", "openai:gpt-4o-mini"
)

T = TypeVar("T")


class Evaluation(BaseModel, Generic[T]):
    model_config = ConfigDict(title="Evaluation")

    reasoning: str
    score: T


@dataclass
class Score:
    probabilities: dict[Any, float]
    reasoning: str = ""

    @cached_property
    def score(self):
        # Most likely category
        if not self.probabilities:
            return None
        return max(self.probabilities, key=self.probabilities.get)  # type: ignore

    @cached_property
    def score_probability(self):
        # Probability of the most likely category
        return self.probabilities[self.score]


"""
ProbabilisticMetric is a class that implement a probabilistic scoring mechanism.

This metric utilizes a prompt to generate responses from a probabilistic model, evaluates the responses based on
the provided scoring criteria, and computes scores along with their associated probabilities. The class is designed
to work with various probabilistic models and allows for customization through parameters such as temperature and model type.

Attention: each class in the scoring function must be a single token.
"""


class ProbabilisticMetric:
    def __init__(
        self,
        name: str,
        prompt: MetricPrompt,
        temperature: float = 1.0,
        model: str = DEFAULT_MODEL,
    ):
        super().__init__()
        self._name = name
        self.prompt = prompt
        self.temperature = temperature
        self.provider = model.split(":")[0]
        self.model = model.split(":")[1]
        if self.provider != "openai":
            raise ValueError(
                f"At the moment, only OpenAI is supported for probabilistic metrics. Got {self.provider}."
            )
        self._client = OpenAI()

        score_type = (
            self.prompt.response_format
            if isinstance(self.prompt.response_format, type)
            else self.prompt.response_format.type  # type: ignore
        )
        self._response_format_type = type(
            "Evaluation", (Evaluation[score_type],), {}
        )
        self._validate()

    def serialize(self):
        return {
            "name": self.name,
            "prompt": self.prompt.serialize(),
            "temperature": self.temperature,
            "model": self.model,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        name = data["name"]
        prompt = MetricPrompt.deserialize(data["prompt"])
        temperature = data["temperature"]
        model = data["model"]
        return cls(
            name=name, prompt=prompt, temperature=temperature, model=model
        )

    @property
    def name(self):
        return self._name

    @property
    def help(self):
        return self.prompt.description or super().help

    @property
    def schema(self) -> Dict[str, MetricField]:
        return {
            f"{self.name}_score": MetricField(
                type=self.prompt.response_format.type  # type: ignore
            ),  # type: ignore
            f"{self.name}_reasoning": MetricField(type=str),
            f"{self.name}_probabilities": MetricField(
                type=Dict[self.prompt.response_format.type, float]  # type: ignore
            ),  # type: ignore
        }

    @property
    def args(self):
        return self.prompt.args

    def _validate(self):
        assert (
            len(self.prompt.get_identifiers()) > 0
        ), "User prompt must have at least one identifier"
        assert self.temperature >= 0, "Temperature must be non-negative"
        assert self.model, "Model must be specified"
        assert "gpt" in self.model, "Model must be a GPT model"

    def _find_token_index(self, tokens, target_token):
        for i, token in enumerate(tokens):
            if token.token.strip() == target_token:
                return i
        return None

    def _process(self, **kwargs) -> Score:
        if self.overloaded_params is not None:
            margs = {
                arg: kwargs[f.name] for arg, f in self.overloaded_params.items()
            }
        else:
            margs = kwargs
        # Init categories
        category_map = {
            str(cat): cat
            for cat in self.prompt.response_format.values()  # type: ignore
        }  # type: ignore
        logprobs = {
            str(cat): -float("inf")
            for cat in self.prompt.response_format.values()  # type: ignore
        }  # type: ignore
        msgs = self.prompt.render(**margs)
        model_response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": msgs["system_prompt"]},
                {"role": "user", "content": msgs["user_prompt"]},
            ],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=len(logprobs),
            response_format=self._response_format_type,
        )
        message = json.loads(model_response.choices[0].message.content)  # type: ignore
        tok_idx = self._find_token_index(
            model_response.choices[0].logprobs.content,  # type: ignore
            str(message.get("score", "")),
        )  # type: ignore
        top_logprobs = (
            model_response.choices[0].logprobs.content[tok_idx].top_logprobs  # type: ignore
        )  # type: ignore
        for logprob in top_logprobs:
            token = logprob.token.strip()
            if token in logprobs:
                logprobs[token] = max(logprobs[token], logprob.logprob)
        probs = {cat: np.exp(lp) for cat, lp in logprobs.items()}
        total_prob = sum(probs.values())
        normalized_probs = {
            category_map[cat]: (prob / total_prob).item()
            for cat, prob in probs.items()
        }
        return Score(
            probabilities=normalized_probs,
            reasoning=message.get("reasoning", ""),
        )

    def __call__(self, **kwargs):
        score = self._process(**kwargs)
        return {
            f"{self.name}_score": score.score,
            f"{self.name}_reasoning": score.reasoning,
            f"{self.name}_probabilities": score.probabilities,
        }


class ProbabilisticMetricFactory:
    @staticmethod
    def create(
        name: str,
        prompt: MetricPrompt,
        temperature: float = 1.0,
        model: str = DEFAULT_MODEL,
    ):
        class_name = name
        ProbabilisticMetricClass = type(
            class_name,
            (ProbabilisticMetric,),
            {
                "__init__": lambda self, **kwargs: super(
                    ProbabilisticMetricClass, self
                ).__init__(
                    class_name,  # type: ignore
                    prompt,
                    temperature,
                    model,  # type: ignore
                )
            },
        )

        return ProbabilisticMetricClass
