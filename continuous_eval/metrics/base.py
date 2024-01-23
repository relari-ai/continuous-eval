from abc import ABC, ABCMeta
from typing import Any, Dict, List

from continuous_eval.llm_factory import DefaultLLM, LLMInterface
from continuous_eval.utils.telemetry import telemetry


class MetricDecoratorMeta(ABCMeta, type):
    def __new__(cls, name, bases, dct):
        for attr, value in dct.items():
            if callable(value) and attr == 'calculate':
                dct[attr] = telemetry.metric_telemetry(value)
            elif callable(value) and attr == 'batch_calculate':
                dct[attr] = telemetry.batch_metric_telemetry(value)
        return type.__new__(cls, name, bases, dct)


class Metric(ABC, metaclass=MetricDecoratorMeta):
    def calculate(self, *args, **kwargs):
        raise NotImplementedError()

    def batch_calculate(self, dataset: List[Dict[str, Any]]):
        return [self.calculate(**datum) for datum in dataset]


class LLMBasedMetric(Metric):
    """
    Base class for all LLM based metrics.
    """

    def __init__(self, model: LLMInterface = DefaultLLM):
        super().__init__()
        assert isinstance(model, LLMInterface), "model must be an instance of LLMInterface."
        self._llm = model
