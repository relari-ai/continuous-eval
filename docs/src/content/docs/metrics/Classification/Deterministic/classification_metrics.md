---
title: Classification Metrics
sidebar:
    order: 1
---

### Definitions

**Classification Metrics** measures the performance of a classification module.

<br>


### Example Usage

Required data items: `class`, `ground_truths`

```python
from continuous_eval.metrics import ClassificationMetrics


datum = {
    "predicted_class": "quantitative_question",
    "ground_truths": ["qualitative_question", "reasoning_question"],
},

metric = ClassificationMetrics()
print(metric(**datum))
```

### Example Output

```JSON
{
    "classification_correctness": 0
}
```

### Aggregate Results

```JSON
{
    "accuracy": 0.8,
    "precision": 0.9,
    "recall": 0.7,
    "f1": 0.8,
}
```