from nltk.tokenize import sent_tokenize

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.retrieval_matching_strategy import MatchingStrategy, is_relevant


class PrecisionRecallF1(Metric):
    def __init__(self, matching_strategy: MatchingStrategy) -> None:
        super().__init__()
        self.matching_strategy = matching_strategy
        assert self.matching_strategy in MatchingStrategy

    def calculate(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate precision, recall and f1 based on different matching strategies.
        # These metrics do not consider the order or rank of relevant information in the retrieval.

        if self.matching_strategy in (
            MatchingStrategy.EXACT_CHUNK_MATCH,
            MatchingStrategy.ROUGE_CHUNK_MATCH,
        ):
            ret_components = retrieved_contexts
            gt_components = ground_truth_contexts

        elif self.matching_strategy in (
            MatchingStrategy.EXACT_SENTENCE_MATCH,
            MatchingStrategy.ROUGE_SENTENCE_MATCH,
        ):
            ret_components = [sentence for chunk in retrieved_contexts for sentence in sent_tokenize(chunk)]
            gt_components = [sentence for chunk in ground_truth_contexts for sentence in sent_tokenize(chunk)]

        relevant_ret_componnets = 0
        hit_gt_components = set()
        for ret_component in ret_components:
            for gt_component in gt_components:
                if is_relevant(ret_component, gt_component, self.matching_strategy):
                    relevant_ret_componnets += 1
                    hit_gt_components.add(gt_component)
                    continue
        precision = relevant_ret_componnets / len(ret_components) if ret_components else 0.0
        recall = len(hit_gt_components) / len(gt_components) if gt_components else 0.0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
