import warnings

import nltk
from rouge import Rouge

from continuous_eval.metrics.base import Metric


# Single Metrics
def _download_punkt():
    nltk.download("punkt", quiet=True)


class TokenOverlap(Metric):
    def __init__(self):
        _download_punkt()
        super().__init__()

    def calculate(self, prediction, reference):
        tokenizer = nltk.tokenize.word_tokenize

        pred_tokens = tokenizer(prediction)
        ref_tokens = tokenizer(reference)

        token_overlap = set(pred_tokens) & set(ref_tokens)
        token_overlap_count = len(token_overlap)

        try:
            precision = token_overlap_count / len(pred_tokens)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = token_overlap_count / len(ref_tokens)
        except ZeroDivisionError:
            recall = 0.0
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0.0

        return {"token_overlap_precision": precision, "token_overlap_recall": recall, "token_overlap_f1": f1}


class RougeScore(Metric):
    def calculate(self, prediction, reference):
        rouge = Rouge()
        if prediction == "":
            rouge_l_score = {"p": 0.0, "r": 0.0, "f": 0.0}
        else:
            scores = rouge.get_scores(prediction, reference)
            rouge_l_score = scores[0]["rouge-l"]
        return {
            "rouge_l_precision": rouge_l_score["p"],
            "rouge_l_recall": rouge_l_score["r"],
            "rouge_l_f1": rouge_l_score["f"],
        }


class BleuScore(Metric):
    def calculate(self, prediction, reference):
        warnings.filterwarnings("ignore")
        bleu = nltk.translate.bleu_score.sentence_bleu([reference], prediction)
        warnings.filterwarnings("default")
        return {"bleu_score": bleu}


# Compound Metrics


class DeterministicFaithfulness(Metric):
    ROUGE_PRECISION_THRESHOLD = 0.5
    TOKEN_OVERLAP_PRECISION_THRESHOLD = 0.5
    BLEU_SCORE_THRESHOLD = 0.5

    def __init__(self):
        _download_punkt()
        super().__init__()

    def calculate(self, answer, retrieved_contexts, **kwargs):
        context = "\n".join(retrieved_contexts)
        sentences = nltk.sent_tokenize(answer)

        rouge_scores = [RougeScore().calculate(sentence, context)["rouge_l_precision"] for sentence in sentences]
        token_overlap_scores = [
            TokenOverlap().calculate(sentence, context)["token_overlap_precision"] for sentence in sentences
        ]
        bleu_scores = [BleuScore().calculate(sentence, context)["bleu_score"] for sentence in sentences]

        rouge_faithfulness = sum(score > self.ROUGE_PRECISION_THRESHOLD for score in rouge_scores) / len(sentences)
        token_overlap_faithfulness = sum(
            score > self.TOKEN_OVERLAP_PRECISION_THRESHOLD for score in token_overlap_scores
        ) / len(sentences)
        bleu_faithfulness = sum(score for score in bleu_scores) / len(sentences)

        return {
            "rouge_faithfulness": rouge_faithfulness,
            "token_overlap_faithfulness": token_overlap_faithfulness,
            "bleu_faithfulness": bleu_faithfulness,
            "rouge_p_by_sentence": rouge_scores,
            "token_overlap_p_by_sentence": token_overlap_scores,
            "blue_score_by_sentence": bleu_scores,
        }


class DeterministicAnswerCorrectness(Metric):
    def calculate(self, answer, ground_truths, **kwargs):
        # calculate the max score across all ground truth answers
        token_scores = [TokenOverlap().calculate(answer, gt_answer) for gt_answer in ground_truths]
        rouge_scores = [RougeScore().calculate(answer, gt_answer) for gt_answer in ground_truths]
        bleu_scores = [BleuScore().calculate(answer, gt_answer) for gt_answer in ground_truths]

        return {
            metric: max(score.get(metric, 0) for score in token_scores + rouge_scores + bleu_scores)
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
