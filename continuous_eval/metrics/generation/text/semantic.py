import warnings
from typing import Dict, List

import pandas as pd

try:
    import torch
except ImportError:
    raise ImportError("To use BertSimilarity, please install PyTorch.")

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.generation.text.bert import BertSimilarity, DebertaScores


class BertAnswerRelevance(Metric):
    def batch(self, answer: List[str], question: List[str]) -> List[Dict[str, float]]:
        score = BertSimilarity().batch(prediction=answer, reference=question)
        return [{"bert_answer_relevance": x} for x in score["bert_similarity"]]

    def __call__(self, answer: str, question: str) -> Dict[str, float]:
        """Measures the semantic similarity between the Generated Answer and the Question"""
        return {"bert_answer_relevance": BertSimilarity()(answer, question)["bert_similarity"]}


class BertAnswerSimilarity(Metric):
    def __call__(self, answer: str, ground_truth_answers: List[str], **kwargs):
        """Measures the semantic similarity between the Generated Answer and the Ground Truth Answers

        Args:
            answer (str): the generated answer
            ground_truth_answers (List[str]): the ground truth answers
        """
        bert_similarity_scores = [BertSimilarity()(answer, gt_answer) for gt_answer in ground_truth_answers]
        return {"bert_answer_similarity": max(score["bert_similarity"] for score in bert_similarity_scores)}

    def _preprocess_dataset(self, answer: List[str], ground_truth_answers: List[List[str]]):
        prediction = list()
        reference = list()
        ids = list()
        for i, (val, ref) in enumerate(zip(answer, ground_truth_answers)):
            for gt_answer in ref:
                prediction.append(val)
                reference.append(gt_answer)
                ids.append(i)
        return prediction, reference, ids

    def batch(self, answer: List[str], ground_truth_answers: List[List[str]]):
        prediction, reference, ids = self._preprocess_dataset(answer, ground_truth_answers)
        score = BertSimilarity().batch(prediction=prediction, reference=reference)
        df = pd.DataFrame({"bert_answer_similarity": score["bert_similarity"], "ids": ids})
        ret = df.groupby("ids").max()
        return [{"bert_answer_similarity": x} for x in ret["bert_answer_similarity"]]


class DebertaAnswerScores(Metric):
    def __init__(self, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
        self.batch_size = 32

    def _ret_keys(self):
        reverse = "reverse_" if self.reverse else ""
        entailment_key = f"deberta_{reverse}answer_entailment"
        contradiction_key = f"deberta_{reverse}answer_contradiction"
        return entailment_key, contradiction_key

    def batch(self, answer: List[str], ground_truth_answers: List[List[str]], **kwargs):
        warnings.filterwarnings("ignore", category=UserWarning)
        entailment_key, contradiction_key = self._ret_keys()
        sentence_pairs = list()
        ids = list()
        for i, (val, ref) in enumerate(zip(answer, ground_truth_answers)):
            for gt_answer in ref:
                if self.reverse:
                    # premise=ground truth => hypothesis=answer
                    sentence_pairs.append((gt_answer, val))
                else:
                    # premise=answer => hypothesis=ground truth
                    sentence_pairs.append((val, gt_answer))
                ids.append(i)

        logits = DebertaScores()(sentence_pairs)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)

        # Group by 'ids' and get the score for the pair with the highest entailment
        df = pd.DataFrame({entailment_key: probs[:, 1], contradiction_key: probs[:, 0], "ids": ids})
        idx = df.groupby("ids")[entailment_key].idxmax()
        return [
            {entailment_key: entailment_value, contradiction_key: contradiction_value}
            for (entailment_value, contradiction_value) in zip(
                df.loc[idx, entailment_key],
                df.loc[idx, contradiction_key],
            )
        ]

    def __call__(self, answer: str, ground_truth_answers: List[str], **kwargs):
        """Measure semantic relationship between the Generated Answer and the Ground Truth Answers in three categories:
        - Entailment: the Generated Answer IMPLIES a Ground Truth Answer.
        - Contradiction: the Generated Answer CONTRADICTS a Ground Truth Answer.
        - Neutral: the Generated Answer and the Ground Truth Answer have neutral logical relationship.

        Args:
            answer (str): the generated answer
            ground_truth_answers (List[str]): the ground truth answers

        Returns:
            _type_: _description_
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        sentence_pairs = list()
        entailment_key, contradiction_key = self._ret_keys()

        for gt_answer in ground_truth_answers:
            if self.reverse:
                # premise=ground truth => hypothesis=answer
                sentence_pairs.append((gt_answer, answer))
            else:
                # premise=answer => hypothesis=ground truth
                sentence_pairs.append((answer, gt_answer))

        logits = DebertaScores()(sentence_pairs)
        # Get the score for the pair with the highest entailment
        logits_with_max_entailment = max(logits, key=lambda sublist: sublist[1])  # type: ignore

        # convert logits into normalized probabilities
        probs = torch.nn.functional.softmax(torch.tensor(logits_with_max_entailment), dim=0)

        return {
            entailment_key: probs[1].item(),
            contradiction_key: probs[0].item(),
        }
