---
title: LLM-based Faithfulness
---

### Definition

**LLM-based Faithfulness** measures how grounded is the generated answer on the retrieved contexts. 

### Example Usage

Required data items: `answer`, and `retrieved_context`

```python
from continuous_eval.metrics.generation.text import Faithfulness

datum = {
    "question": "Who wrote 'Romeo and Juliet'?",
    "retrieved_context": ["William Shakespeare is the author of 'Romeo and Juliet'."],
    "answer": "Shakespeare wrote 'Romeo and Juliet'",
    "ground_truth_answers": "Shakespeare",
}
metric = Faithfulness()
print(metric(**datum))
```

### Sample Output

```python
{
    "faithfulness": 1.0,
    "reasoning": "The statement directly reflects the context.",
}
```
