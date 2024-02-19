from abc import ABC, ABCMeta
from typing import Any, Dict, List, Optional

from continuous_eval.llm_factory import DefaultLLM, LLMInterface
from continuous_eval.utils.telemetry import telemetry

# from continuous_eval.eval.manager import eval_manager
# from continuous_eval.eval.dataset import DatasetField


# class MetricDecoratorMeta(ABCMeta, type):
#     def __new__(cls, name, bases, dct):
#         for attr, value in dct.items():
#             if callable(value) and attr == 'calculate':
#                 dct[attr] = telemetry.metric_telemetry(value)
#             elif callable(value) and attr == 'batch_calculate':
#                 dct[attr] = telemetry.batch_metric_telemetry(value)
#         return type.__new__(cls, name, bases, dct)


class Metric(ABC):  # , metaclass=MetricDecoratorMeta):
    def __init__(self) -> None:
        super().__init__()
        self._overloaded_params = None

    def use(self, **kwargs) -> "Metric":
        self._overloaded_params = kwargs
        return self

    @property
    def overloaded_params(self):
        return self._overloaded_params

    def __call__(self, **kwargs):
        # Implement this method in the subclass
        raise NotImplementedError()

    def batch(self, **kwargs) -> Any:
        kwargs_ = [{key: kwargs[key][i] for key in kwargs} for i in range(len(next(iter(kwargs.values()))))]
        return [self(**kw) for kw in kwargs_]

    @property
    def name(self):
        return self.__class__.__name__


class LLMBasedMetric(Metric):
    """
    Base class for all LLM based metrics.
    """

    def __init__(self, model: Optional[LLMInterface] = None):
        super().__init__()
        if model is None:
            self._llm = DefaultLLM()
        else:
            self._llm = model
        assert isinstance(self._llm, LLMInterface), "model must be an instance of LLMInterface."
