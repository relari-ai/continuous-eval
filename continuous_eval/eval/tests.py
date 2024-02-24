from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Test(ABC):
    @property
    def name(self) -> str:
        """Name of the test."""
        raise NotImplementedError

    @abstractmethod
    def run(self, metrics_per_sample) -> bool:
        """Run the test on the module.

        Args:
            module (_type_): _description_

        Returns:
            bool: Pass (True) or Fail (False)
        """
        raise NotImplementedError


# Some common tests
class GreaterOrEqualThan(Test):
    def __init__(self, test_name: str, metric_name: str, min_value: float):
        self._name = test_name
        self._key = metric_name
        self._value = min_value

    @property
    def name(self) -> str:
        return self._name

    def run(self, metrics_per_sample: List[Dict[str, Any]]) -> bool:
        return all(sample[self._key] >= self._value for sample in metrics_per_sample)


class MeanGreaterOrEqualThan(Test):
    def __init__(self, test_name: str, metric_name: str, min_value: float):
        self._name = test_name
        self._key = metric_name
        self._value = min_value

    @property
    def name(self) -> str:
        return self._name

    def run(self, metrics_per_sample: List[Dict[str, Any]]) -> bool:
        return sum(sample[self._key] for sample in metrics_per_sample) / len(metrics_per_sample) >= self._value
