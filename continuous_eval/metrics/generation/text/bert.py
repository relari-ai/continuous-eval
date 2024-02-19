from typing import Dict, List

import torch
from sentence_transformers import CrossEncoder
from transformers import BertModel, BertTokenizer


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
