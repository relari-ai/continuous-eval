from typing import Any, Dict

from continuous_eval.llms import LLMFactory
from continuous_eval.metrics.base import Field as MetricField
from continuous_eval.metrics.base import Metric, MetricPrompt
from continuous_eval.metrics.base.response_type import ScoringFunction


class LLMMetric(Metric):
    def __init__(
        self,
        name: str,
        prompt: MetricPrompt,
        temperature: float = 1.0,
        model: str = LLMFactory.default(),
    ):
        super().__init__()
        if isinstance(prompt.response_format, type):
            assert issubclass(
                prompt.response_format, ScoringFunction
            ), "Prompt response_format must be a ScoringFunction"
        else:
            assert isinstance(
                prompt.response_format, ScoringFunction
            ), "Prompt response_format must be a ScoringFunction"
        self._name = name
        self.prompt = prompt
        self.temperature = temperature
        self.model = model
        self._llm = LLMFactory.get(model)

    @property
    def name(self):
        return self._name

    @property
    def help(self):
        return self.prompt.description or "No description available"

    @property
    def schema(self) -> Dict[str, MetricField]:
        if hasattr(self.prompt.response_format, "schema"):
            return {
                k: MetricField(type=t)
                for k, t in self.prompt.response_format.schema.items()
            }  # type: ignore
        else:
            return {
                f"{self.name}_score": MetricField(
                    type=self.prompt.response_format.type
                ),  # type: ignore
                f"{self.name}_reasoning": MetricField(type=str),
            }

    @property
    def args(self):
        return self.prompt.args

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

    def __call__(self, **kwargs):
        if self.overloaded_params is not None:
            margs = {
                arg: kwargs[f.name] for arg, f in self.overloaded_params.items()
            }
        else:
            margs = kwargs
        prompt = self.prompt.render(**margs)
        res = self._llm.run(prompt=prompt, temperature=self.temperature)
        score = self.prompt.response_format.score(res)  # type: ignore
        return score


class LLMMetricFactory:
    @staticmethod
    def create(
        name: str,
        prompt: MetricPrompt,
        temperature: float = 1.0,
        model: str = LLMFactory.default(),
    ):
        class_name = name
        ProbabilisticMetricClass = type(
            class_name,
            (LLMMetric,),
            {
                "__init__": lambda self, **kwargs: super(
                    ProbabilisticMetricClass, self
                ).__init__(class_name, prompt, temperature, model)  # type: ignore
            },
        )

        return ProbabilisticMetricClass
