from typing import Dict, List

try:
    import torch
except ImportError:
    raise ImportError("To use BertSimilarity, please install PyTorch.")
try:
    from sentence_transformers import CrossEncoder
    from transformers import BertModel, BertTokenizer
except ImportError:
    raise ImportError("To use BertSimilarity, please install sentence-transformers and transformers.")
from continuous_eval.metrics.base import Metric


class DebertaScores:
    def __init__(self):
        self._model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
        self._batch_size = 32

    @property
    def device(self):
        return self._model._target_device

    def _batch_predict(self, sentence_pairs, batch_size):
        """
        Predicts in batches.
        """
        batched_predictions = []
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i : i + batch_size]
            predictions = self._model.predict(batch)
            batched_predictions.extend(predictions)
        return batched_predictions

    def __call__(self, sentence_pairs, batch_size=32):
        """
        Splits sentence_pairs into batches of size batch_size and performs prediction on each batch.
        """
        return self._batch_predict(sentence_pairs, batch_size)


class BertSimilarity(Metric):
    def __init__(self, pooler_output: bool = False):
        super().__init__()
        # Load pre-trained BERT model and tokenizer
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model = BertModel.from_pretrained("bert-base-uncased")
        self._pooler_output = pooler_output
        self.batch_size = 32

    def batch(self, prediction: List[str], reference: List[str]):
        # Function to yield batches of data
        def mini_batches(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]

        # Process batches
        all_similarities = []
        for pred_batch, ref_batch in zip(
            mini_batches(prediction, self.batch_size),
            mini_batches(reference, self.batch_size),
        ):
            batch_result = self._subprocess(pred_batch, ref_batch)
            all_similarities.extend(batch_result)
        return {"bert_similarity": all_similarities}

    def _subprocess(self, prediction: List[str], reference: List[str]):
        predictions = self._tokenizer(prediction, padding=True)
        references = self._tokenizer(reference, padding=True)

        # Get BERT embeddings for the tokens
        with torch.no_grad():
            pred_embedding = self._model(  # type: ignore
                torch.tensor(predictions["input_ids"]),
                attention_mask=torch.tensor(predictions["attention_mask"]),
            )
            ref_embedding = self._model(  # type: ignore
                torch.tensor(references["input_ids"]),
                attention_mask=torch.tensor(references["attention_mask"]),
            )
            if self._pooler_output:
                pred_embedding = pred_embedding.pooler_output
                ref_embedding = ref_embedding.pooler_output
            else:
                pred_embedding = pred_embedding[0].mean(dim=1)
                ref_embedding = ref_embedding[0].mean(dim=1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        semantic_similarity = cosine_similarity(pred_embedding.T, ref_embedding.T)
        semantic_similarity = torch.clip(semantic_similarity, min=0.0, max=1.0)
        return semantic_similarity.tolist()

    def __call__(self, prediction: str, reference: str):
        res = self.batch(prediction=[prediction], reference=[reference])
        return {"bert_similarity": res["bert_similarity"][0]}
