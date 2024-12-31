from dataclasses import dataclass
from typing import Dict, List, Union

import nltk

from continuous_eval.metrics.base import Arg, Field, Metric
from continuous_eval.metrics.generation.text.utils import (
    BleuScore,
    RougeScore,
    TokenOverlap,
)


@dataclass(frozen=True)
class DeterministicFaithfulnessConfig:
    rouge_precision_threshold: float = 0.5
    token_overlap_precision_threshold: float = 0.5


class DeterministicFaithfulness(Metric):
    """
    Evaluate the faithfulness of the generated text with respect to the retrieved context.
    """

    def __init__(
        self,
        thresholds: DeterministicFaithfulnessConfig = DeterministicFaithfulnessConfig(),
    ):
        super().__init__(is_cpu_bound=True)
        self.cfg = thresholds
        self._token_overlap = TokenOverlap()
        self._rouge = RougeScore()
        self._bleu = BleuScore()

    def compute(
        self, answer: str, retrieved_context: Union[List[str], str], **kwargs
    ):
        """Computes the faithfulness of the answer with respect to the retrieved contexts."""
        if isinstance(retrieved_context, str):
            retrieved_context = [retrieved_context]

        context = "\n".join(retrieved_context)
        sentences = nltk.sent_tokenize(answer)

        rouge_scores = [
            self._rouge.calculate(sentence, context)["rouge_l_precision"]
            for sentence in sentences
        ]
        token_overlap_scores = [
            self._token_overlap.calculate(sentence, context)[
                "token_overlap_precision"
            ]
            for sentence in sentences
        ]
        bleu_scores = [
            self._bleu.calculate(sentence, context)["bleu_score"]
            for sentence in sentences
        ]

        rouge_faithfulness = sum(
            score >= self.cfg.rouge_precision_threshold
            for score in rouge_scores
        ) / len(sentences)
        token_overlap_faithfulness = sum(
            score >= self.cfg.token_overlap_precision_threshold
            for score in token_overlap_scores
        ) / len(sentences)
        bleu_faithfulness = sum(score for score in bleu_scores) / len(sentences)

        return {
            "rouge_faithfulness": rouge_faithfulness,
            "token_overlap_faithfulness": token_overlap_faithfulness,
            "bleu_faithfulness": bleu_faithfulness,
            "rouge_p_by_sentence": rouge_scores,
            "token_overlap_p_by_sentence": token_overlap_scores,
            "bleu_score_by_sentence": bleu_scores,
        }

    @property
    def args(self):
        return {
            "answer": Arg(type=str, description="The generated text"),
            "retrieved_context": Arg(
                type=List[str], description="The retrieved context"
            ),
        }

    @property
    def schema(self):
        return {
            "rouge_faithfulness": Field(type=float),
            "token_overlap_faithfulness": Field(type=float),
            "bleu_faithfulness": Field(type=float),
            "rouge_p_by_sentence": Field(type=List[float]),
            "token_overlap_p_by_sentence": Field(type=List[float]),
            "bleu_score_by_sentence": Field(type=List[float]),
        }


class DeterministicAnswerCorrectness(Metric):
    """Evaluates the correctness of the answer with respect to the ground truth."""

    def __init__(self):
        super().__init__(is_cpu_bound=True)

    def compute(
        self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs
    ):
        """Computes the correctness of the answer with respect to the ground truth."""
        # calculate the max score across all ground truth answers
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]
        token_scores = [
            TokenOverlap().calculate(answer, gt_answer)
            for gt_answer in ground_truth_answers
        ]
        rouge_scores = [
            RougeScore().calculate(answer, gt_answer)
            for gt_answer in ground_truth_answers
        ]
        bleu_scores = [
            BleuScore().calculate(answer, gt_answer)
            for gt_answer in ground_truth_answers
        ]

        return {
            metric: max(
                score.get(metric, 0)
                for score in token_scores + rouge_scores + bleu_scores
            )
            for metric in [
                "rouge_l_recall",
                "rouge_l_precision",
                "rouge_l_f1",
                "token_overlap_recall",
                "token_overlap_precision",
                "token_overlap_f1",
                "bleu_score",
            ]
        }

    @property
    def schema(self) -> Dict[str, Field]:
        return {
            "rouge_l_recall": Field(type=float, limits=(0, 1)),
            "rouge_l_precision": Field(type=float, limits=(0, 1)),
            "rouge_l_f1": Field(type=float, limits=(0, 1)),
            "token_overlap_recall": Field(type=float, limits=(0, 1)),
            "token_overlap_precision": Field(type=float, limits=(0, 1)),
            "token_overlap_f1": Field(type=float, limits=(0, 1)),
            "bleu_score": Field(type=float, limits=(0, 1)),
        }


class DeterministicMatch(Metric):
    """Evaluates the exact match score between the generated text and the ground truth."""

    def __init__(self):
        super().__init__(is_cpu_bound=True)

    def _cmp(self, ans, gt):
        ans_ = ans.strip().lower() if isinstance(ans, str) else ans
        gt_ = gt.strip().lower() if isinstance(gt, str) else gt
        try:
            return float(type(gt)(ans_) == gt_)
        except ValueError:
            return 0.0

    def compute(
        self, answer: str, ground_truth_answers: Union[str, List[str]], **kwargs
    ):
        """Computes the correctness of the answer with respect to the ground truth."""
        if not isinstance(ground_truth_answers, list):
            ground_truth_answers = [ground_truth_answers]
        match_scores = [self._cmp(answer, x) for x in ground_truth_answers]
        return {"match": max(match_scores)}

    @property
    def schema(self) -> Dict[str, Field]:
        return {"match": Field(type=float, limits=(0, 1))}


class FleschKincaidReadability(Metric):
    """Computes the Flesch Reading Ease and Flesch-Kincaid Grade Level readability scores."""

    def __init__(self):
        super().__init__(is_cpu_bound=True)
        self._word_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        self._syl_tokenizer = nltk.tokenize.SyllableTokenizer()

    def compute(self, answer, **kwargs):
        if answer.strip() == "":
            return {
                "flesch_reading_ease": 121.22,
                "flesch_kincaid_grade_level": 0.0,
            }
        # Calculate the number of sentences, words, and syllables.
        words = self._word_tokenizer.tokenize(answer)
        num_syllables = sum(
            len(self._syl_tokenizer.tokenize(word)) for word in words
        )
        num_sentences = len(nltk.sent_tokenize(answer))
        num_words = len(words)
        # Flesch Reading-Ease score
        try:
            fre_score = (
                206.835
                - 1.015 * (num_words / num_sentences)
                - 84.6 * (num_syllables / num_words)
            )
        except ZeroDivisionError:
            fre_score = 121.22
        # Flesch–Kincaid Grade Level
        try:
            fk_grade = (
                0.39 * (num_words / num_sentences)
                + 11.8 * (num_syllables / num_words)
                - 15.59
            )
        except ZeroDivisionError:
            fk_grade = 0.0
        return {
            "flesch_reading_ease": fre_score,
            "flesch_kincaid_grade_level": fk_grade,
        }

    @property
    def schema(self) -> Dict[str, Field]:
        return {
            "flesch_reading_ease": Field(type=float),
            "flesch_kincaid_grade_level": Field(type=float),
        }
