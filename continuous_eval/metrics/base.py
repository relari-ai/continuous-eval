import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dotenv import load_dotenv

from continuous_eval.llm_factory import LLMFactory

load_dotenv()

EVAL_LLM = os.getenv("EVAL_LLM", "gpt-3.5-turbo-1106")


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

    def __init__(self, model=EVAL_LLM):
        super().__init__()
        self.llm_factory = LLMFactory(model)
