import torch
from transformers import BertModel, BertTokenizer
from sentence_transformers import CrossEncoder

from continuous_eval.metrics.base import Metric

# Single Metrics

class BertSimilarity:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def calculate(self, prediction, reference):
        # Tokenize the prediction and reference texts
        pred_tokens = self.tokenizer.tokenize(prediction)
        ref_tokens = self.tokenizer.tokenize(reference)

        # Convert tokens to token IDs
        pred_ids = self.tokenizer.convert_tokens_to_ids(pred_tokens)
        ref_ids = self.tokenizer.convert_tokens_to_ids(ref_tokens)

        # Convert token IDs to tensors
        pred_tensor = torch.tensor([pred_ids])
        ref_tensor = torch.tensor([ref_ids])

        # Get BERT embeddings for the tokens
        with torch.no_grad():
            pred_embedding = self.model(pred_tensor)[0].squeeze(0).mean(dim=0)
            ref_embedding = self.model(ref_tensor)[0].squeeze(0).mean(dim=0)

        # Calculate cosine similarity between the embeddings
        cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        semantic_similarity = cosine_similarity(pred_embedding, ref_embedding).item()
        semantic_similarity = max(0.0, min(semantic_similarity, 1.0))  # clip in [0, 1]
        return {"bert_similarity": semantic_similarity}

class BertAnswerRelevance(Metric):
    def calculate(self, answer, question, **kwargs):
        return {
            "bert_answer_relevance": BertSimilarity().calculate(answer, question)[
                "bert_similarity"
            ]
        }


class BertAnswerSimilarity(Metric):
    def calculate(self, answer, ground_truths, **kwargs):
        bert_similarity_scores = [
            BertSimilarity().calculate(answer, gt_answer) for gt_answer in ground_truths
        ]
        return {
            "bert_answer_similarity": max(
                score["bert_similarity"] for score in bert_similarity_scores
            )
        }
    
class DebertaScores:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
    
    def calculate(self, sentence_pairs):
        scores = self.model.predict(sentence_pairs)
        return scores

class DebertaAnswerScores(Metric):
    
    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def calculate(self, answer, ground_truths, **kwargs):
        sentence_pairs = []

        for gt_answer in ground_truths:
            if self.reverse:
                # premise=ground truth => hypothesis=answer
                sentence_pairs.append((gt_answer, answer))
            else:
                # premise=answer => hypothesis=ground truth
                sentence_pairs.append((answer, gt_answer))

        scores = DebertaScores().calculate(sentence_pairs)

        # Get the score for the pair with the highest entailment
        scores_with_max_entailment = max(scores, key=lambda sublist: sublist[1])

        if self.reverse:
            return {
                "deberta_reverse_answer_entailment": scores_with_max_entailment[1],
                "deberta_reverse_answer_contradiction": scores_with_max_entailment[0],
            }
        else:
            return {
                "deberta_answer_entailment": scores_with_max_entailment[1],
                "deberta_answer_contradiction": scores_with_max_entailment[0],
            }