from abc import ABC, ABCMeta
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import pandas as pd

from continuous_eval.llm_factory import DefaultLLM, LLMInterface
from continuous_eval.utils.telemetry import telemetry


class MetricDecoratorMeta(ABCMeta, type):
    def __new__(cls, name, bases, dct):
        for attr, value in dct.items():
            if callable(value) and attr == '__call__':
                dct[attr] = telemetry.metric_telemetry(value)
            elif callable(value) and attr == 'batch':
                pass
                # dct[attr] = telemetry.batch_metric_telemetry(value)
        return type.__new__(cls, name, bases, dct)


class Metric(ABC, metaclass=MetricDecoratorMeta):
    def __init__(self) -> None:
        super().__init__()
        self._overloaded_params = None
        self.max_workers = 32

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
        if self.max_workers <= 1:
            return [self(**kw) for kw in kwargs_]
        instances = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_instances = [executor.submit(lambda kw: self(**kw), kw) for kw in kwargs_]
            for future in future_instances:
                instances.append(future.result())
        return instances

    def aggregate(self, results: List[Any]) -> Any:
        # Default implementation
        sanitize = lambda results: [{k: v for k, v in r.items() if not isinstance(v, (list, str))} for r in results]
        agg = pd.DataFrame(sanitize(results))
        return agg.mean().to_dict()

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
