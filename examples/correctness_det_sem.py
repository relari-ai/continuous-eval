from time import perf_counter

import numpy as np
import pandas as pd

from continuous_eval.classifiers import EnsembleMetric
from continuous_eval.classifiers.utils import eval_prediction
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.datatypes import DataSplit, SplitRatios
from continuous_eval.evaluators import GenerationEvaluator
from continuous_eval.metrics import DebertaAnswerScores, DeterministicAnswerCorrectness

dataset = example_data_downloader("correctness")
dataset = dataset[dataset["annotation"] != "refuse-to-answer"]

tic = perf_counter()
evaluator = GenerationEvaluator(
    dataset=dataset,
    metrics=[
        DeterministicAnswerCorrectness(),
        DebertaAnswerScores(),
    ],
)
evaluator.run()
toc = perf_counter()
print(f"Evaluation completed in {toc - tic:.2f}s")
print(set(evaluator.aggregated_results))

evaluator.save("det_sem.jsonl")  # Save for future use

X = pd.DataFrame(evaluator.results)
y = dataset["annotation"].map({"correct": 1, "incorrect": 0}).astype(int).to_numpy()

datasplit = DataSplit(
    X=X,
    y=y,
    split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
    features=[
        "token_recall",
        "deberta_answer_entailment",
        "deberta_answer_contradiction",
    ],
    oversample=True,
)
clf = EnsembleMetric(training=datasplit.train, calibration=datasplit.calibration)
y_hat, y_set = clf.predict(datasplit.test.X)
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")
