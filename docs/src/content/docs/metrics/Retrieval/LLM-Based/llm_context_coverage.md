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


### Example Usage

Required data items: `question`, `retrieved_context`, `ground_truth_answers`

```python
from continuous_eval.metrics import LLMBasedContextCoverage
from continuous_eval.llm_factory import LLMFactory

datum = {
    "question": "What is the largest and second city in France?",
    "retrieved_contexts": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truth_contexts": ["Marseille is the second largest city in France.", "Paris is the largest city in the country." ],
    "answer": "Paris",
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
                "statement_2": "Lyon is the second largest city in France.",
                "reason": "The context does not provide information about the ranking of Lyon in terms of size compared to other French cities.",
                "Attributed": 0
            }
        ]
    }
}
```