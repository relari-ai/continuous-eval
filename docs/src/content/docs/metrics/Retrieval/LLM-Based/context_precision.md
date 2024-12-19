---
title: LLM-based Context Precision
---

### Definition

Context Precision is used to measure information density.

$$
\text{Context Precision} =
\frac{
  \text{Number of Relevant Chunks in Retrieved Contexts}
}{
  \text{Total Number of Chunks in Retrieved Contexts}
}
$$

while

$$
\text{Mean Average Precision (mAP)} = \frac{1}{\text{Number of Relevant Chunks}} \sum_{j=1}^{\text{Number of Retrieved Context}} \text{ Precision at Rank } j
$$


### Example Usage

Required data items: `question`, `retrieved_context`

```python
from continuous_eval.metrics.retrieval import ContextPrecision

datum = {
    "question": "What is the capital of France?",
    "retrieved_context": [
        "Paris is the capital of France and also the largest city in the country.",
        "Lyon is a major city in France.",
    ],
}

metric = ContextPrecision()
print(metric(**datum))
```

### Sample Output

```python
{
    "percentage_relevant": 0.5,
    "context_precision": 0.5000000000746547,
    "context_mean_average_precision": 1.0,
}
```
