import warnings
from typing import Any, Dict, List

import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import BertModel, BertTokenizer

from continuous_eval.metrics.base import Metric

# Single Metrics


class DebertaScores:
    def __init__(self):
        self._model = CrossEncoder("cross-encoder/nli-deberta-v3-large")

    @property
    def device(self):
        return self._model._target_device

    def calculate(self, sentence_pairs):
        return self._model.predict(sentence_pairs)


class BertSimilarity:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model = BertModel.from_pretrained("bert-base-uncased")

    def _tokenize(self, data: List[Dict[str, str]]):
        predictions = self._tokenizer([datum["prediction"] for datum in data], padding=True)
        references = self._tokenizer([datum["reference"] for datum in data], padding=True)
        return predictions, references

    def batch_calculate(self, data: List[Dict[str, str]], pooler_output: bool = False):
        predictions, references = self._tokenize(data)

        # Get BERT embeddings for the tokens
        with torch.no_grad():
            pred_embedding = self._model(
                torch.tensor(predictions["input_ids"]),
                attention_mask=torch.tensor(predictions["attention_mask"]),
            )
            ref_embedding = self._model(
                torch.tensor(references["input_ids"]),
                attention_mask=torch.tensor(references["attention_mask"]),
            )
            if pooler_output:
                pred_embedding = pred_embedding.pooler_output
                ref_embedding = ref_embedding.pooler_output
            else:
                pred_embedding = pred_embedding[0].mean(dim=1)
                ref_embedding = ref_embedding[0].mean(dim=1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        semantic_similarity = cosine_similarity(pred_embedding.T, ref_embedding.T)
        semantic_similarity = torch.clip(semantic_similarity, min=0.0, max=1.0)
        return {"bert_similarity": semantic_similarity.tolist()}

    def calculate(self, prediction: str, reference: str, pooler_output: bool = False):
        res = self.batch_calculate(
            [{"prediction": prediction, "reference": reference}],
            pooler_output=pooler_output,
        )
        return {"bert_similarity": res["bert_similarity"][0]}


class BertAnswerRelevance(Metric):
    def _preprocess_dataset(self, dataset: List[Dict[str, Any]]):
        return [{"prediction": datum["answer"], "reference": datum["question"]} for datum in dataset]

    def batch_calculate(self, dataset: List[Dict[str, Any]]):
        data = self._preprocess_dataset(dataset)
        score = BertSimilarity().batch_calculate(data)
        return [{"bert_answer_relevance": x} for x in score["bert_similarity"]]

    def calculate(self, answer, question, **kwargs):
        return {"bert_answer_relevance": BertSimilarity().calculate(answer, question)["bert_similarity"]}


class BertAnswerSimilarity(Metric):
    def calculate(self, answer, ground_truths, **kwargs):
        bert_similarity_scores = [BertSimilarity().calculate(answer, gt_answer) for gt_answer in ground_truths]
        return {"bert_answer_similarity": max(score["bert_similarity"] for score in bert_similarity_scores)}

    def _preprocess_dataset(self, dataset: List[Dict[str, Any]]):
        data = list()
        ids = list()
        for i, datum in enumerate(dataset):
            for gt_answer in datum["ground_truths"]:
                data.append({"prediction": datum["answer"], "reference": gt_answer})
                ids.append(i)
        return data, ids

    def batch_calculate(self, dataset: List[Dict[str, Any]]):
        data, ids = self._preprocess_dataset(dataset)
        score = BertSimilarity().batch_calculate(data)
        df = pd.DataFrame({"bert_answer_similarity": score["bert_similarity"], "ids": ids})
        ret = df.groupby("ids").max()
        return [{"bert_answer_similarity": x} for x in ret["bert_answer_similarity"]]


class DebertaAnswerScores(Metric):
    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def _ret_keys(self):
        reverse = "reverse_" if self.reverse else ""
        entailment_key = f"deberta_{reverse}answer_entailment"
        contradiction_key = f"deberta_{reverse}answer_contradiction"
        return entailment_key, contradiction_key

    def batch_calculate(self, dataset: List[Dict[str, Any]]):
        warnings.filterwarnings("ignore", category=UserWarning)
        entailment_key, contradiction_key = self._ret_keys()
        sentence_pairs = list()
        ids = list()
        for i, datum in enumerate(dataset):
            for gt_answer in datum["ground_truths"]:
                if self.reverse:
                    # premise=ground truth => hypothesis=answer
                    sentence_pairs.append((gt_answer, datum["answer"]))
                else:
                    # premise=answer => hypothesis=ground truth
                    sentence_pairs.append((datum["answer"], gt_answer))
                ids.append(i)

        logits = DebertaScores().calculate(sentence_pairs)
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

    def calculate(self, answer, ground_truths, **kwargs):
        warnings.filterwarnings("ignore", category=UserWarning)
        sentence_pairs = list()
        entailment_key, contradiction_key = self._ret_keys()

        for gt_answer in ground_truths:
            if self.reverse:
                # premise=ground truth => hypothesis=answer
                sentence_pairs.append((gt_answer, answer))
            else:
                # premise=answer => hypothesis=ground truth
                sentence_pairs.append((answer, gt_answer))

        logits = DebertaScores().calculate(sentence_pairs)
        # Get the score for the pair with the highest entailment
        logits_with_max_entailment = max(logits, key=lambda sublist: sublist[1])

        # convert logits into normalized probabilities
        probs = torch.nn.functional.softmax(torch.tensor(logits_with_max_entailment), dim=0)

        return {
            entailment_key: probs[1].item(),
            contradiction_key: probs[0].item(),
        }
