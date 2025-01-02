---
title: Context Coverage
---

### Definition

Context Coverage measures completeness of the retrieved contexts to generated a ground truth answer.

$$
\text{Context Coverage} =
\frac{
  \text{Number of Statements in Generated Answer Attributed to the Ground Truth Contexts}
}{
  \text{Total Number of Statements in Generated Answer}
}
$$

This metric requires the LLM evaluator to output correct and complex JSON. If the JSON cannot be parsed, the score returns -1.0.

### Example Usage

Required data items: `question`, `retrieved_context`, `ground_truths`

```python
from continuous_eval.metrics.retrieval import ContextCoverage

datum = {
    "question": "What is the largest and second city in France?",
    "retrieved_context": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truth_answers": ["Paris is the largest city in France and Marseille is the second largest."],
}

metric = ContextCoverage()
print(metric(**datum))
```

### Sample Output

```JSON
{
    "context_coverage": 0.5,
    "statements": [
        "Paris is the largest city in France.",
        "Marseille is the second largest city in France.",
    ],
}
```
