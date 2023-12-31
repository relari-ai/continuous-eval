---
title: Context Coverage
---

### Definition

Inputs: `question`, `retrieved_context` and `ground_truth_answers`

$$
\text{LLM-Based Context Coverage} =
\frac{
  \text{Number of Statements in Generated Answer Attributed to the Ground Truth Contexts}
}{
  \text{Total Number of Statements in Generated Answer}
}
$$

this metric is used to measure completeness of the retrieved contexts to generated a ground truth answer.

### Example Usage

```python
from continuous_eval.metrics import LLMBasedContextCoverage

metric = LLMBasedContextCoverage(
    model = "gpt-4-1105-preview", 
    use_few_shot: bool = True,
    log_relevance_by_context: bool = False
)
```

### Sample Output

```JSON
{
    "LLM_based_context_coverage": 0.5,
    "LLM_based_context_statements": {
        "classification": [
            {
                "statement_1": "Reginald Dewayne Grimes played for the New England Patriots during the 2000 New England Patriots season.",
                "reason": "The context does not mention Reginald Dewayne Grimes, so this cannot be attributed to the given context.",
                "Attributed": 0
            },
            {
                "statement_2": "The 2000 New England Patriots season was the 31st season for the team in the National Football League.",
                "reason": "This is directly stated in the context.",
                "Attributed": 1
            }
        ]
    },
}
```