---
title: Classification Metrics
sidebar:
    order: 1
---

### Definitions

**Classification Match** measures the performance of a classification module. It measures the match of individual predictions and also calculates aggregate results over a dataset.

<br>


### Example Usage

Required data items: `predicted_class`, `ground_truth_class`

```python
from continuous_eval.metrics import ClassificationMatch


datum = {
    "predicted_class": "quantitative_question",
    "ground_truth_class": "qualitative_question",
},

metric = ClassificationMatch()
print(metric(**datum))
```

### Example Output

```JSON
{
    "predicted_class": "quantitative_question",
    "ground_truth_class":"qualitative_question",
    "class_correct": 0
}
```

### Aggregate Results

```JSON
{
    "cls_accuracy": accuracy,
    "cls_precision": precision,
    "cls_recall": recall,
    "cls_f1": f1,
}
```