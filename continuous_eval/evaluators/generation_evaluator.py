import logging as logger
from typing import List

import pandas as pd
from tqdm import tqdm

from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.metrics import DeterministicFaithfulness, Metric
from continuous_eval.dataset import Dataset

class GenerationEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[Metric] = [DeterministicFaithfulness()],
    ):
        super().__init__(dataset=dataset, metrics=metrics)

    def run(self):
        self._results = self._calculate_metrics()

    def _calculate_metrics(self):
        results = []
        for _, row in tqdm(
            self.dataset.iterrows(),
            total=len(self.dataset),
            desc="Examples evaluated",
        ):
            datum = row.to_dict()
            result = dict()
            for metric in self.metrics:
                try:
                    result.update(metric.calculate(**datum))
                except Exception as e:
                    logger.warning(f"Error calculating {metric}. Skipping...")
                    print(e)
                    continue
            results.append(result)

        return results
