from typing import List

import pandas as pd
from tqdm import tqdm

from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.evaluators.utils import validate_dataset
from continuous_eval.metrics import (
    MatchingStrategy,
    Metric,
    PrecisionRecallF1,
)


class RetrievalEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: List[Metric] = [PrecisionRecallF1(MatchingStrategy.EXACT_CHUNK_MATCH)],
    ):
        super().__init__(metrics)

    def run(
        self,
        dataset: List[dict],
        k: int = None,
        aggregate=True,
    ):
        validate_dataset(dataset)
        if k is not None and k < 0:
            print(
                "K must be a positive integer. Leave it as None if not used (consider all retrieved chunks in metrics)."
            )

        results = self._calculate_metrics(dataset, k)

        if aggregate:
            results_df = pd.DataFrame(results)
            return results_df.mean().to_dict()
        else:
            return results

    def _calculate_metrics(self, dataset, k):
        results = []
        for datum in tqdm(
            dataset, total=len(dataset), desc="Calculating retrieval metrics"
        ):
            if k is None:
                item = datum
            else:
                item = datum.copy()
                item["retrieved_contexts"] = datum["retrieved_contexts"][:k]

            result = {}
            for metric in self.metrics:
                try:
                    result.update(metric.calculate(**item))
                except Exception as e:
                    print(f"Error calculating {metric}. Skipping...")
                    print(e)
                    continue
            results.append(result)
        return results
