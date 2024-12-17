from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from continuous_eval.llms import LLMFactory
from continuous_eval.metrics.base.llm import LLMMetric
from continuous_eval.metrics.base.metric import Arg, Field
from continuous_eval.metrics.base.prompt import MetricPrompt
from continuous_eval.metrics.base.response_type import JSON

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
        self.prompt = MetricPrompt(
            sys_prompt,
            user_prompt,
            response_format=JSON(
                {k: v.type_hint for k, v in response_format.items()}
            ),
        )
        super().__init__(
            name=name, prompt=self.prompt, temperature=temperature, model=model
        )
