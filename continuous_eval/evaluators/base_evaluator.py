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

    @staticmethod
    def _sanitize_pre_aggregate(results):
        return [
            {k: v for k, v in r.items() if not isinstance(v, (list, str))}
            for r in results
        ]
