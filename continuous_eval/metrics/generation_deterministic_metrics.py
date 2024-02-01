import warnings

import nltk
from nltk import download as nltk_download
from rouge import Rouge

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.utils.simple_tokenizer import SimpleTokenizer

# Single Metrics


class TokenOverlap(Metric):
    def __init__(self):
        super().__init__()
        self._tokenizer = SimpleTokenizer()

    def _tokenize(self, text, language="english"):
        sentences = nltk.tokenize.sent_tokenize(text, language)
        return [token for sent in sentences for token in self._tokenizer.tokenize(sent)]

    def calculate(self, prediction, reference):

        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

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

        return {
            "token_overlap_precision": precision,
            "token_overlap_recall": recall,
            "token_overlap_f1": f1,
        }


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
        nltk_download("punkt", quiet=True)
        super().__init__()

    def calculate(self, answer, retrieved_contexts, **kwargs):
        context = "\n".join(retrieved_contexts)
        sentences = nltk.sent_tokenize(answer)

        rouge_scores = [RougeScore().calculate(sentence, context)["rouge_l_precision"] for sentence in sentences]
        token_overlap_scores = [
            TokenOverlap().calculate(sentence, context)["token_overlap_precision"] for sentence in sentences
        ]
        bleu_scores = [BleuScore().calculate(sentence, context)["bleu_score"] for sentence in sentences]

        rouge_faithfulness = sum(score >= self.ROUGE_PRECISION_THRESHOLD for score in rouge_scores) / len(sentences)
        token_overlap_faithfulness = sum(
            score >= self.TOKEN_OVERLAP_PRECISION_THRESHOLD for score in token_overlap_scores
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


class FleschKincaidReadability(Metric):
    def __init__(self):
        super().__init__()
        self._word_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        self._syl_tokenizer = nltk.tokenize.SyllableTokenizer()

    def calculate(self, answer, **kwargs):
        if answer.strip() == "":
            return {
                "flesch_reading_ease": 121.22,
                "flesch_kincaid_grade_level": 0.0,
            }
        # Calculate the number of sentences, words, and syllables.
        words = self._word_tokenizer.tokenize(answer)
        num_syllables = sum(len(self._syl_tokenizer.tokenize(word)) for word in words)
        num_sentences = len(nltk.sent_tokenize(answer))
        num_words = len(words)
        # Flesch Reading-Ease score
        fre_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        # Flesch–Kincaid Grade Level
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        return {
            "flesch_reading_ease": fre_score,
            "flesch_kincaid_grade_level": fk_grade,
        }
