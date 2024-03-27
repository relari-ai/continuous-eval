import re
import string
import warnings
from typing import Optional, Union

import nltk
from rouge import Rouge

from continuous_eval.metrics._utils.simple_tokenizer import SimpleTokenizer


def _numeric_matcher(input_val, min_val, max_val) -> Optional[float]:
    pattern = r"\d+(?:\.\d+)?"  # Match any number (integer or float)
    matches = re.findall(pattern, input_val)
    if not matches:
        return None
    return max(min_val, min(max_val, float(matches[0])))


class ScoringFunctions:
    @staticmethod
    def Numeric(
        min_val: Union[int, float] = 1,
        max_val: Union[int, float] = 5,
    ):
        assert min_val < max_val, "min_val should be less than max_val"
        return lambda input_val: _numeric_matcher(input_val, min_val, max_val)

    @staticmethod
    def Identity(value: str):
        return value


class TokenOverlap:
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


class RougeScore:
    def calculate(self, prediction, reference):
        rouge = Rouge()
        if prediction == "" or all(c in string.punctuation for c in prediction):
            # If the prediction is empty or only punctuation, the ROUGE score is 0
            rouge_l_score = {"p": 0.0, "r": 0.0, "f": 0.0}
        else:
            scores = rouge.get_scores(prediction, reference)
            rouge_l_score = scores[0]["rouge-l"]
        return {
            "rouge_l_precision": rouge_l_score["p"],
            "rouge_l_recall": rouge_l_score["r"],
            "rouge_l_f1": rouge_l_score["f"],
        }


class BleuScore:
    def calculate(self, prediction, reference):
        warnings.filterwarnings("ignore")
        bleu = nltk.translate.bleu_score.sentence_bleu([reference], prediction)
        warnings.filterwarnings("default")
        return {"bleu_score": bleu}
