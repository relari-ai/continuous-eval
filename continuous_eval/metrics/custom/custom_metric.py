from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from continuous_eval.llms import LLMFactory
from continuous_eval.metrics.base import (
    Arg,
    Field,
    MetricPrompt,
    response_type,
)
from continuous_eval.metrics.base.llm import LLMMetric
from continuous_eval.metrics.base.probabilistic import ProbabilisticMetric

_CWD = Path(__file__).parent


@dataclass(frozen=True, eq=True)
class Example:
    input: Dict[str, Any]
    output: Dict[str, Any]


class CustomMetric(LLMMetric):
    def __init__(
        self,
        name: str,
        criteria: str,
        rubric: str,
        arguments: Dict[str, Arg],
        response_format: Dict[str, Field],
        examples: Optional[List[Example]] = None,
        temperature: float = 1.0,
        model: str = LLMFactory.default(),
    ):
        with open(_CWD / "custom_metric_sys.jinja2") as f:
            raw_system_prompt = f.read()
        with open(_CWD / "custom_metric_user.jinja2") as f:
            raw_user_prompt = f.read()
        env = Environment(loader=BaseLoader())
        sys_prompt_template = env.from_string(raw_system_prompt)
        user_prompt_template = env.from_string(raw_user_prompt)
        sys_prompt = sys_prompt_template.render(
            criteria=criteria,
            rubric=rubric,
            examples=examples,
            response_format=response_format,
        )
        user_prompt = user_prompt_template.render(arguments=arguments)
        self._criteria = criteria
        self.prompt = MetricPrompt(
            sys_prompt,
            user_prompt,
            response_format=response_type.JSON(
                {k: v.type for k, v in response_format.items()}
            ),
        )
        super().__init__(
            name=name, prompt=self.prompt, temperature=temperature, model=model
        )

    @property
    def help(self):
        return self._criteria


class ProbabilisticCustomMetric(ProbabilisticMetric):
    def __init__(
        self,
        name: str,
        criteria: str,
        rubric: str,
        arguments: Dict[str, Arg],
        response_format: response_type.ResponseFormatBaseType,
        examples: Optional[List[Example]] = None,
        temperature: float = 1.0,
        model: str = LLMFactory.default(),
    ):
        if not isinstance(
            response_format, response_type.ResponseFormatBaseType
        ):
            raise ValueError("response_format must be a ResponseFormatBaseType")
        if isinstance(response_format, response_type.JSON):
            raise ValueError(
                "Probabilistic metrics do not support JSON response format, use CustomMetric instead"
            )
        with open(_CWD / "custom_metric_sys_probabilistic.jinja2") as f:
            raw_system_prompt = f.read()
        with open(_CWD / "custom_metric_user.jinja2") as f:
            raw_user_prompt = f.read()
        env = Environment(loader=BaseLoader())
        sys_prompt_template = env.from_string(raw_system_prompt)
        user_prompt_template = env.from_string(raw_user_prompt)
        sys_prompt = sys_prompt_template.render(
            criteria=criteria,
            rubric=rubric,
            examples=examples,
            response_format=response_format,
        )
        user_prompt = user_prompt_template.render(arguments=arguments)
        self._criteria = criteria
        self.prompt = MetricPrompt(
            sys_prompt,
            user_prompt,
            response_format=response_format,
        )
        super().__init__(
            name=name, prompt=self.prompt, temperature=temperature, model=model
        )

    @property
    def help(self):
        return self._criteria
