from abc import ABC, abstractmethod
from typing import List


class BaseEvaluator(ABC):
    @abstractmethod
    def __init__(self, metrics: List[str]):
        self.metrics = metrics

    @abstractmethod
    def run(self, dataset, aggregate=True):
        pass

    @abstractmethod
    def _calculate_metrics(self, dataset):
        pass
