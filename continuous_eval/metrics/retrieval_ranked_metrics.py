import logging as logger
from math import log

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.retrieval_matching_strategy import MatchingStrategy, is_relevant


class RankedRetrievalMetrics(Metric):
    def __init__(self, matching_strategy: MatchingStrategy) -> None:
        super().__init__()
        self.matching_strategy = matching_strategy
        assert self.matching_strategy in MatchingStrategy

    def calculate(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate ranked metrics (MAP, MRR, NDCG) based on different matching strategies.

        # Rank metrics only calculated at chunk level, converting sentence-level matching strategies to chunk-level
        if self.matching_strategy == MatchingStrategy.EXACT_SENTENCE_MATCH:
            self.matching_strategy = MatchingStrategy.EXACT_CHUNK_MATCH
            logger.warning(
                "Ranked metrics are calculated at chunk level, "
                f"using {self.matching_strategy} strategy for Ranked Metrics."
            )

        if self.matching_strategy == MatchingStrategy.ROUGE_SENTENCE_MATCH:
            self.matching_strategy = MatchingStrategy.ROUGE_CHUNK_MATCH
            logger.warning(
                "Ranked metrics are calculated at chunk level, "
                f"using {self.matching_strategy} strategy for Ranked Metrics."
            )

        map = self.calculate_average_precision(retrieved_contexts, ground_truth_contexts)
        mrr = self.calculate_reciprocal_rank(retrieved_contexts, ground_truth_contexts)
        ndcg = self.calculate_normalized_discounted_cumulative_gain(retrieved_contexts, ground_truth_contexts)

        return {"Average Precision": map, "Mean Reciprocal Rank": mrr, "NDCG": ndcg}

    def calculate_average_precision(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate average precision for a single query retrieval

        # Calculate average precision for each relevant chunk
        average_precision = 0
        relevant_chunks = 0

        for i, chunk in enumerate(retrieved_contexts):
            for ground_truth_chunk in ground_truth_contexts:
                if is_relevant(chunk, ground_truth_chunk, self.matching_strategy):
                    relevant_chunks += 1
                    average_precision += relevant_chunks / (i + 1)
                    continue

        return average_precision / relevant_chunks if relevant_chunks else 0

    def calculate_reciprocal_rank(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate reciprocal rank for a single query retrieval

        # Calculate reciprocal rank for each relevant chunk
        for i, chunk in enumerate(retrieved_contexts):
            for ground_truth_chunk in ground_truth_contexts:
                if is_relevant(chunk, ground_truth_chunk, self.matching_strategy):
                    return 1 / (i + 1)

        # If no relevant chunk is found, return 0
        return 0

    def calculate_normalized_discounted_cumulative_gain(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate normalized discounted cumulative gain for a single query retrieval

        # Calculate discounted cumulative gain
        dcg = 0
        for i, chunk in enumerate(retrieved_contexts):
            for ground_truth_chunk in ground_truth_contexts:
                if is_relevant(chunk, ground_truth_chunk, self.matching_strategy):
                    # Calculate relevance score (relevant gain = 1)
                    dcg += 1 / log(i + 2, 2)
                    continue

        # Calculate ideal discounted cumulative gain
        idcg = 0
        for i in range(len(ground_truth_contexts)):
            idcg += 1 / log(i + 2, 2)

        return dcg / idcg
