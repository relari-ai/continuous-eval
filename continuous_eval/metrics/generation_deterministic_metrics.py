import nltk
import torch
from rouge import Rouge
from transformers import BertModel, BertTokenizer
import warnings

from continuous_eval.metrics.base import Metric

# Single Metrics


class TokenOverlap(Metric):
    def calculate(self, prediction, reference):
        tokenizer = nltk.tokenize.word_tokenize

        pred_tokens = tokenizer(prediction)
        ref_tokens = tokenizer(reference)

        token_overlap = set(pred_tokens) & set(ref_tokens)
        token_overlap_count = len(token_overlap)

        precision = token_overlap_count / len(pred_tokens)
        recall = token_overlap_count / len(ref_tokens)
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0

        return {"token_precision": precision, "token_recall": recall, "token_f1": f1}


class RougeScore(Metric):
    def calculate(self, prediction, reference):
        rouge = Rouge()
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


# Compound Metrics


class DeterministicFaithfulness(Metric):
    ROUGE_PRECISION_THRESHOLD = 0.5
    TOKEN_OVERLAP_PRECISION_THRESHOLD = 0.5

    def calculate(self, answer, retrieved_contexts, **kwargs):
        context = "\n".join(retrieved_contexts)
        sentences = nltk.sent_tokenize(answer)

        rouge_scores = [
            RougeScore().calculate(sentence, context)["rouge_l_precision"]
            for sentence in sentences
        ]
        token_overlap_scores = [
            TokenOverlap().calculate(sentence, context)["token_precision"]
            for sentence in sentences
        ]
        bleu_scores = [
            BleuScore().calculate(sentence, context)["bleu_score"]
            for sentence in sentences
        ]

        rouge_faithfulness = sum(
            score > self.ROUGE_PRECISION_THRESHOLD for score in rouge_scores
        ) / len(sentences)
        token_overlap_faithfulness = sum(
            score > self.TOKEN_OVERLAP_PRECISION_THRESHOLD
            for score in token_overlap_scores
        ) / len(sentences)
        avg_sentence_bleu = sum(score for score in bleu_scores) / len(sentences)
        min_sentence_bleu = min(bleu_scores)

        return {
            "rouge_faithfulness": rouge_faithfulness,
            "token_overlap_faithfulness": token_overlap_faithfulness,
            "avg_sentence_bleu": avg_sentence_bleu,
            "min_sentence_bleu": min_sentence_bleu,
            "rouge_scores_p_by_sentence": rouge_scores,
            "token_overlap_p_by_sentence": token_overlap_scores,
        }


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


class DeterministicAnswerRelevance(Metric):
    def calculate(self, answer, ground_truths, **kwargs):
        # calculate the max score across all ground truth answers
        token_scores = [
            TokenOverlap().calculate(answer, gt_answer) for gt_answer in ground_truths
        ]
        rouge_scores = [
            RougeScore().calculate(answer, gt_answer) for gt_answer in ground_truths
        ]
        bleu_scores = [
            BleuScore().calculate(answer, gt_answer) for gt_answer in ground_truths
        ]
        return {
            metric: max(
                score.get(metric, 0)
                for score in token_scores + rouge_scores + bleu_scores
            )
            for metric in set().union(*token_scores, *rouge_scores, *bleu_scores)
        }
