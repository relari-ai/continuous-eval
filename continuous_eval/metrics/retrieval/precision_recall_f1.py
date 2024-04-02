from typing import List

from nltk.tokenize import sent_tokenize

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.retrieval.matching_strategy import MatchingStrategy, MatchingStrategyType, RougeChunkMatch


class PrecisionRecallF1(Metric):
    def __init__(self, matching_strategy: MatchingStrategy = RougeChunkMatch()):
        super().__init__()
        assert isinstance(
            matching_strategy, MatchingStrategy
        ), "Matching strategy must be an instance of MatchingStrategy."
        self.matching_strategy = matching_strategy

    def __call__(self, retrieved_context: List[str], ground_truth_context: List[str], **kwargs):
        """
        Calculate the precision, recall, and f1 score for the given datapoint.
        """

        # Calculate precision, recall and f1 based on different matching strategies.
        # These metrics do not consider the order or rank of relevant information in the retrieval.
        if self.matching_strategy.type == MatchingStrategyType.CHUNK_MATCH:
            ret_components = retrieved_context
            gt_components = ground_truth_context
        elif self.matching_strategy.type == MatchingStrategyType.SENTENCE_MATCH:
            ret_components = [sentence for chunk in retrieved_context for sentence in sent_tokenize(chunk)]
            gt_components = [sentence for chunk in ground_truth_context for sentence in sent_tokenize(chunk)]

        relevant_ret_components = 0
        hit_gt_components = set()
        gt_components = set(gt_components)  # remove duplicates in ground truth context if any
        for ret_component in ret_components:
            ret_component_matched = False
            for gt_component in gt_components:
                if self.matching_strategy.is_relevant(ret_component, gt_component):
                    ret_component_matched = True
                    hit_gt_components.add(gt_component)
                    continue
            relevant_ret_components += int(ret_component_matched)
        precision = relevant_ret_components / len(ret_components) if ret_components else 0.0
        recall = len(hit_gt_components) / len(gt_components) if gt_components else 0.0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return {
            "context_precision": precision,
            "context_recall": recall,
            "context_f1": f1,
        }
