from time import perf_counter

import numpy as np
import pandas as pd

from continuous_eval.classifiers import EnsembleMetric
from continuous_eval.classifiers.utils import eval_prediction
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.datatypes import DataSplit, SplitRatios
from continuous_eval.llm_factory import LLMFactory
from continuous_eval.metrics import LLMBasedAnswerCorrectness

# # Download the correctness dataset and remove examples where the LLL refused to answer (i.e., said "I don't know")
dataset = example_data_downloader("correctness")
dataset = dataset[dataset["annotation"] != "refuse-to-answer"]

# Let's assume you already run the evaluator (see examples ensemble_metric.py) and saved the results
results = pd.read_json(path_or_buf="det_sem.jsonl", lines=True)

# Now let's use the results to train a classifier to predict the correctness of the answers (as evaluated by a human annotator)
X = pd.DataFrame(results)
y = dataset["annotation"].map({"correct": 1, "incorrect": 0}).astype(int).to_numpy()

# We split the dataset into train, test, and calibration sets
#
datasplit = DataSplit(
    X=X,
    y=y,
    dataset=dataset,
    split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
    features=[
        "token_overlap_recall",
        "deberta_answer_entailment",
        "deberta_answer_contradiction",
    ],
    oversample=True,
)

# We use the train and calibration sets to train the classifier
llm_metric = LLMBasedAnswerCorrectness(LLMFactory("gpt-4-1106-preview"))


def judicator(idx):
    # The judicator receives the index of the example in the test set where the classifier is undecided
    # and in this case, since we are computing the correctness of the sample,
    # it returns True if the example is correct and False otherwise
    datum = datasplit.test_full.X.iloc[idx].to_dict()
    return llm_metric.calculate(**datum)["LLM_based_answer_correctness"] >= 0.5


# Let's train a metric ensamble classifier
clf = EnsembleMetric(training=datasplit.train, calibration=datasplit.calibration)

# We run the classifier on the test set without the judicator
print("Without judicator")
tic = perf_counter()
y_hat, y_set = clf.predict(datasplit.test.X)
toc = perf_counter()
print(f"Prediction completed in {(toc - tic)*1000:.2f}ms")
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")

# We run the classifier again on the same set, but this time with the judicator
print("\nWith judicator")
tic = perf_counter()
y_hat, y_set = clf.predict(datasplit.test.X, judicator=judicator)
toc = perf_counter()
print(f"Prediction completed in {toc - tic:.2f}s")
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")
