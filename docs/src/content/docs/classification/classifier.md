---
title: Classifier
description: Use metric to predict human evaluator.
---

```python
# Load dataset
dataset = Dataset.from_jsonl("data/correctness.jsonl")

# Remove "I don't know" answers
dataset = dataset[dataset["annotation"] != "refuse-to-answer"]

# Load precomputed features (see evaluators)
X = pd.read_json("eval_correctness.jsonl", lines=True)
y = dataset["annotation"].map({"correct": 1, "incorrect": 0}).astype(int)

# Let's split the dataset into training, calibration and test
# We also oversample the training set to have a balanced dataset
datasplit = DataSplit(
    X=X,
    y=y,
    split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
    features=["token_recall"],
    oversample=True,
)

# Train a classifier
clf = ConformalClassifier(
    training=datasplit.train, calibration=datasplit.calibration
)

# Use the classifier to predict the test set
y_hat, y_set = clf.predict(datasplit.test.X)

# Evaluate the predictions
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")
```
