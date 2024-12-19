---
title: Classification Metrics
sidebar:
    order: 1
---

### Definitions

**Single Label Classification** measures the performance of a classification module.

<br>

### Example Usage

Required data items: `predicted_class`, `ground_truth_class`

```python
from continuous_eval.metrics.classification import SingleLabelClassification

datum = {
    "predicted_class": "quantitative_question",
    "ground_truth_class": "qualitative_question",
}

metric = SingleLabelClassification(classes={"qualitative_question", "quantitative_question"})
print(metric(**datum))
```

#### Example Output

```JSON
{
  'classification_prediction': 'quantitative_question', 'classification_ground_truth': 'qualitative_question', 'classification_correct': False
}
```

### Aggregate Results

```python
from continuous_eval.metrics.classification import SingleLabelClassification

y_pred = ["A", "A", "B", "A", "B"]
y_true =  ["A", "B", "B", "A", "B"]

metric = SingleLabelClassification(classes={"A", "B"})
results = [metric(y, y_gt) for y, y_gt in zip(y_pred, y_true)]
print(metric.aggregate(results))
```

:::note
The evaluation manager will aggregate the results of the metric automatically, calling `aggregate` is only necessary if you are not using the evaluation manager.
:::

#### Example Output

```JSON
{
  "accuracy": 0.8,
  "balanced_accuracy": 0.8333333333333333,
  "precision": 0.8333333333333333,
  "recall": 0.8333333333333333,
  "f1": 0.8,
}
```
