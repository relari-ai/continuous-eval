---
title: LLM-based Context Coverage
---

### Definition

Context Coverage measures completeness of the retrieved contexts to generated a ground truth answer.


$$
\text{LLM-Based Context Coverage} =
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
from continuous_eval.metrics import LLMBasedContextCoverage
from continuous_eval.llm_factory import LLMFactory

datum = {
    "question": "What is the largest and second city in France?",
    "retrieved_contexts": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truths": ["Paris is the largest city in France and Marseille is the second largest."],
}

metric = LLMBasedContextCoverage(LLMFactory("gpt-4-1106-preview"))
print(metric.calculate(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_context_coverage': 0.5, 
    'LLM_based_context_statements':
    {
        "classification": [
            {
                "statement_1": "Paris is the largest city in France.",
                "reason": "This is directly stated in the context.",
                "Attributed": 1
            },
            {
                "statement_2": "Marseille is the second largest city in France.",
                "reason": "This information is not provided in the context, which only mentions Paris and Lyon.",
                "Attributed": 0
            }
        ]
    }
}
```