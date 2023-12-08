from typing import List

import pandas as pd
from tqdm import tqdm

from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.evaluators.utils import validate_dataset
from continuous_eval.metrics import (
    BertAnswerRelevance,
    BertAnswerSimilarity,
    DeterministicAnswerRelevance,
    LLMBasedAnswerCorrectness,
    LLMBasedFaithfulness,
    Metric,
    RougeSentenceFaithfulness,
)


class GenerationEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: List[Metric] = [RougeSentenceFaithfulness()],
    ):
        super().__init__(metrics)
        self.metrics = metrics

    def run(self, dataset, use_few_shot: bool = True, aggregate=True):
        validate_dataset(dataset)
        results = self._calculate_metrics(dataset, use_few_shot)

        if aggregate:
            results_df = pd.DataFrame(results)
            return results_df.mean().to_dict()
        else:
            return results

    def _calculate_metrics(self, dataset, use_few_shot):
        results = []
        for item in tqdm(
            dataset, total=len(dataset), desc="Calculating generation metrics"
        ):
            question = item["question"]
            retrieval = item["retrieved_contexts"]
            answer = item["answer"]
            ground_truth_answers = item["ground_truths"]

            result = {}
            for metric in self.metrics:
                if isinstance(metric, RougeSentenceFaithfulness):
                    context = "\n".join(retrieval)
                    result.update(
                        metric.calculate(
                            answer=answer,
                            context=context,
                        )
                    )
                elif isinstance(metric, DeterministicAnswerRelevance):
                    result.update(
                        metric.calculate(
                            answer=answer,
                            ground_truth_answers=ground_truth_answers,
                        )
                    )
                elif isinstance(metric, BertAnswerSimilarity):
                    result.update(
                        metric.calculate(
                            answer=answer,
                            ground_truth_answers=ground_truth_answers,
                        )
                    )
                elif isinstance(metric, BertAnswerRelevance):
                    result.update(
                        metric.calculate(
                            answer=answer,
                            question=question,
                        )
                    )
                elif isinstance(metric, LLMBasedFaithfulness):
                    result.update(
                        metric.calculate(
                            retrieved_context=retrieval,
                            answer=answer,
                            use_few_shot=use_few_shot,
                        )
                    )
                elif isinstance(metric, LLMBasedAnswerCorrectness):
                    result.update(
                        metric.calculate(
                            question=question,
                            answer=answer,
                            ground_truth_answers=ground_truth_answers,
                            use_few_shot=use_few_shot,
                        )
                    )
                else:
                    print(f"Unsupported metric: {metric}. Skipping...")
                    continue
            results.append(result)

        return results
