from abc import ABC, abstractmethod
from typing import Any, Dict, List

from continuous_eval.llm_factory import DefaultLLM, LLMInterface


class Metric(ABC):
    @abstractmethod
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
