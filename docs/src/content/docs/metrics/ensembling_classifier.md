---
title: Metric Ensembling
sidebar:
  badge:
    text: beta
    variant: tip
---


The aim of ensembling different metrics to predict the human label is to combine the strengths and balance out the weaknesses of individual metrics, ultimately leading to more accurate, robust, and reliable predictions. 

Each metric might capture different aspects of the data or be sensitive to different patterns, so when we combine them, we often get a more comprehensive view.

## What is Conformal Prediction?

Conformal Prediction is a statistical technique that quantifies the confidence level of a prediction.
In this case, we are trying to predict whether the answer is correct (or faithful).
With conformal prediction, instead of just saying “yes” (or “no”), the model tells us “the answer is correct with probability at least 90%”.
In essence, conformal prediction doesn’t just give you an answer; it tells you how confident you can be in that answer.
If the model is uncertain, conformal prediction will tell you it’s “undecided”. 
For the undecided datapoints, we ask the more powerful GPT-4 to judge its correctness.

## Metric Ensembling

The `MetricEnsemble` class helps you to ensemble multiple metrics to predict a ground truth label, such us human labels.
The class leverage the conformal prediction technique to compute a reliable 

Parameters:

- `training: XYData`: training data, it should contain `training.X` (the metrics output, also referred as _features_) and `training.Y` (the ground truth label)
- `calibration: XYData`: as before but used for the calibration of the conformal predictor
- `alpha: float`: significance level, default to 0.1. The significance level os the probability that a prediction will not be included in the predicted set, serving as a measure of the confidence or reliability of the prediction. For example if alpha is 0.1, then the prediction set will contain the correct label with probability 0.9.
- `random_state: Optional[int]`: random seed, default to None

The `MetricEnsemble` class has the following methods:

- `predict(self, X: pd.DataFrame, judicator: Optional[Callable] = None)`: it takes as input a dataframe of metrics output and returns a dataframe of predictions

The `predict` returns two numpy vectors:

- `y_hat` a binary (1/0) vector with best-effort predictions of the ensemble
- `y_set` a binary array of size (N, 2) where the first column is 1 is, for the significance level set by `alpha`, the sample can be classified as negative, and the second column is 1 if the sample can be classified as positive.

The set prediction (`y_set`) can have both columns set to 1, meaning that the ensemble is undecided.
This happen because the particular choice of metrics in the ensemble is not confident enough or the significance level is too high.
In such cases the `predict` method will call the `judicator` function (if not `None`) to make a final decision.

The `judicator` function takes as input the index of the sample where the predictor is undecided and must return a boolean value (True/False) indicating the final decision.

### Example

In this exampel we want to use deterministic and semantic metrics to predict the correctness of the answers (as evaluated by a human annotator).
When these two metrics alone are not sufficient to produce a confident prediction, we use the LLM to make the final decision.

As first thing we compute the deterministic and semantic metrics:

```python
from pathlib import Path

import numpy as np

from continuous_eval.classifiers import EnsembleMetric
from continuous_eval.classifiers.utils import eval_prediction
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.datatypes import DataSplit, SplitRatios
from continuous_eval.eval import Dataset, SingleModulePipeline
from continuous_eval.eval.manager import eval_manager
from continuous_eval.llm_factory import LLMFactory
from continuous_eval.metrics.generation.text import (
    DebertaAnswerScores,
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
    LLMBasedAnswerCorrectness,
)


# Download the correctness dataset and remove examples where the LLL refused to answer (i.e., said "I don't know")
dataset_jsonl = example_data_downloader("correctness")
dataset = Dataset(dataset_jsonl)
dataset.filter(lambda x: x["annotation"] != "refuse-to-answer")

# Let's define the system under evaluation
# We use a single module pipeline to evaluate the correctness of the answers
# using the DeterministicAnswerCorrectness metric, the DebertaAnswerScores metric, and the FleschKincaidReadability metric
# Attention: the DebertaAnswerScores metric requires the DeBERTa model (slow on CPU)
pipeline = SingleModulePipeline(
    dataset=dataset,
    eval=[
        DeterministicAnswerCorrectness().use(
            answer=dataset.answer, ground_truth_answers=dataset.ground_truths  # type: ignore
        ),
        DebertaAnswerScores().use(answer=dataset.answer, ground_truth_answers=dataset.ground_truths),  # type: ignore
        FleschKincaidReadability().use(answer=dataset.answer),  # type: ignore
    ],
)

# We start the evaluation manager and run the metrics
eval_manager.set_pipeline(pipeline)
eval_manager.evaluation.results = dataset.data
eval_manager.run_metrics()
eval_manager.metrics.save(Path("metrics_results.json"))
```

We now split the samples in train, test, and calibration sets and train the classifier.
Note that we are using only the `"token_overlap_recall"`,`"deberta_answer_entailment"`, and `"deberta_answer_contradiction"` to train the classifier. 

```python
# Now we building the data for the ensemble classifier
# X is the input the classifier can use
# y is the target the classifier should predict (1 for correct, 0 for incorrect)
X = eval_manager.metrics.to_pandas()
y = map(lambda x: 1 if x == "correct" else 0, dataset["annotation"])

# We split the data into train, test, and calibration sets
# We also specify the features we want to use for the classifier
# We also oversample the train set to balance the classes
datasplit = DataSplit(
    X=X,
    y=y,
    split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
    features=[
        "token_overlap_recall",
        "deberta_answer_entailment",
        "deberta_answer_contradiction",
        "flesch_reading_ease",
    ],
    oversample=True,
)

# We use the train and calibration sets to train the classifier
predictor = EnsembleMetric(training=datasplit.train, calibration=datasplit.calibration)
```

Finally we run the classifier and evaluate the results:

```python
# We then use the test set to evaluate the classifier
print("Running predictor (without judicator)")
y_hat, y_set = predictor.predict(datasplit.test.X)
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")
```

The output would be something like:

```text
Prediction completed in 2.36ms
{'precision': 0.9627329192546584, 'recall': 0.824468085106383, 'f1': 0.8882521489971348, 'accuracy': 0.8340425531914893}
Undecided: 61 (25.96%)
```

#### Using a judicator

Let's assume we want to use the LLM to make the final decision when the classifier is undecided.
We can define a `judicator` function that takes as input the index of the sample where the classifier is undecided and returns a boolean value (True/False) indicating the final decision.

```python
# We use the train and calibration sets to train the classifier
print("\nRunning predictor (with judicator)")
llm_metric = LLMBasedAnswerCorrectness(LLMFactory("gpt-4-1106-preview"))


def judicator(idx):
    # The judicator receives the index of the example in the test set where the classifier is undecided
    # and in this case, since we are computing the correctness of the sample,
    # it returns True if the example is correct and False otherwise
    datum = dataset.data[idx]
    metric_result = llm_metric(
        question=datum["question"],
        answer=datum["answer"],
        ground_truth_answers=datum["ground_truths"],
    )
    return metric_result["LLM_based_answer_correctness"] >= 0.5
```

To use the judicator we simply pass it to the `predict` method:

```python
y_hat, _ = predictor.predict(datasplit.test.X, judicator=judicator)
print(eval_prediction(datasplit.test.y, y_hat))
```

The output would be something like:

```text
Prediction completed in 245.73s
{'precision': 0.9818181818181818, 'recall': 0.8617021276595744, 'f1': 0.9178470254957507, 'accuracy': 0.8765957446808511}
Undecided: 0 (0.00%)
```

Here the `predict` function called the LLM in the _25.96%_ of the cases where the classifier was undecided.
The classifier is no longer undecided and the performance improved but the prediction time increased from _2.36ms_ to _245.73s_.
