from pathlib import Path
import pandas as pd
import numpy as np
from continuous_eval.datatypes import DataSplit, SplitRatios
from continuous_eval.classifiers import HybridClassifier
from continuous_eval.classifiers.utils import eval_prediction

CORRECTNESS_COLS = [
    "token_recall",
    "token_precision",
    "token_f1",
    "rouge_l_recall",
    "rouge_l_precision",
    "rouge_l_f1",
    "bleu_score",
    "bert_answer_relevance",
    "bert_answer_similarity",
    "deberta_contradiction",
    "deberta_entailment",
    "deberta_neutral",
    "deberta_reverse_entailment",
    "deberta_symmetric_entailment",
    "Gemini_based_answer_correctness",
    "GPT4_based_answer_correctness",
    "GPT35_based_answer_correctness",
    "Claude_based_answer_correctness",
]

FAITHFULNESS_COLS = [
    "question",
    "answer",
    "ground_truths",
    "rouge_faithfulness",
    "token_overlap_faithfulness",
    "avg_sentence_bleu",
    "min_sentence_bleu",
    "max_rouge_scores",
    "max_token_overlap",
    "avg_token_overlap",
    "avg_rouge_score",
    "Gemini_based_binary_faithfulness",
    "Gemini_based_by_statement_faithfulness",
    "GPT4_based_binary_faithfulness",
    "GPT4_based_by_statement_faithfulness",
    "GPT35_based_binary_faithfulness",
    "GPT35_based_by_statement_faithfulness",
]


def load_correctness_data(csv_path: Path):
    # fmt: off
    df = pd.read_csv(csv_path)
    df = df[df["annotation"] != "refuse-to-answer"]
    df = df[df["answer"].str.contains("I don't know").fillna(False)==False]
    df.loc[:, "encoded_annotation"] = df["annotation"].map({"correct": 1, "incorrect": 0})
    features = df[CORRECTNESS_COLS].copy()
    features.loc[:, "target"] = df["encoded_annotation"]
    features = features.dropna()
    # fmt: on
    return features[CORRECTNESS_COLS], features["target"].to_numpy()


def load_faithfulness_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["is_passage_relevant"] == True]
    df.loc[:, "encoded_annotation"] = df.loc[:, "faithfulness_annotation"].map(
        {"completely": 1, "partially": 0.5, "not-from-passage": 0}
    )
    df.loc[:, "encoded_binary_annotation"] = df.loc[:, "faithfulness_annotation"].map(
        {"completely": 1, "partially": 0, "not-from-passage": 0}
    )

    mean_over_list = lambda s: np.mean(eval(s)) if isinstance(s, str) else np.nan
    max_over_list = lambda s: np.max(eval(s)) if isinstance(s, str) else np.nan
    df.loc[:, "avg_token_overlap"] = df.loc[:, "token_overlap_p_by_sentence"].apply(
        mean_over_list
    )
    df.loc[:, "avg_rouge_score"] = df.loc[:, "rouge_scores_p_by_sentence"].apply(
        mean_over_list
    )
    df.loc[:, "max_rouge_scores"] = df.loc[:, "rouge_scores_p_by_sentence"].apply(
        max_over_list
    )
    df.loc[:, "max_token_overlap"] = df.loc[:, "token_overlap_p_by_sentence"].apply(
        max_over_list
    )
    df = df.dropna()
    return (
        df[FAITHFULNESS_COLS],
        df["encoded_annotation"].to_numpy(),
    )


def main():
    features = [
        "token_recall",
        "rouge_l_recall",
        "deberta_entailment",
        "deberta_contradiction",
        "deberta_reverse_entailment",
    ]
    dataset = DataSplit(
        *load_correctness_data(Path("data/correctness_master.csv")),
        split_ratios=SplitRatios(train=0.7, test=0.1, calibration=0.2),
        features=features,
        oversample=True
    )
    clf = HybridClassifier(training=dataset.train, calibration=dataset.calibration)
    y_hat, y_set = clf.predict(dataset.test.X)
    print(eval_prediction(dataset.test.y, y_hat))


if __name__ == "__main__":
    main()
