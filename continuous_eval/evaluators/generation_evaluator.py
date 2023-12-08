from typing import List

import pandas as pd
from tqdm import tqdm

from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.evaluators.utils import validate_dataset
from continuous_eval.metrics import (
    Metric,
    RougeSentenceFaithfulness,
)
import logging as logger

class GenerationEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: List[Metric] = [RougeSentenceFaithfulness()],
    ):
        super().__init__(metrics)
        self.metrics = metrics

    def run(self, dataset, aggregate:bool=True):
        validate_dataset(dataset)
        results = self._calculate_metrics(dataset)

        if aggregate:
            results_df = pd.DataFrame(results)
            return results_df.mean().to_dict()
        else:
            return results

    def _calculate_metrics(self, dataset):
        results = []
        for item in tqdm(
            dataset, total=len(dataset), desc="Calculating generation metrics"
        ):
            result = {}
            for metric in self.metrics:
                try:
                    result.update(metric.calculate(**item))
                except Exception as e:
                    logger.warning(f"Error calculating {metric}. Skipping...")
                    print(e)
                    continue
            results.append(result)

        return results
