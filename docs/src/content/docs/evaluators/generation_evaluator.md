---
title: Generation Evaluator
---

```python
from continuous_eval.evaluators import GenerationEvaluator
from continuous_eval.dataset import Dataset
from continuous_eval.metrics import DeterministicAnswerRelevance

dataset = Dataset.from_jsonl("data/correctness.jsonl")
evaluator = GenerationEvaluator(
    dataset=dataset,
    metrics=[
        DeterministicAnswerRelevance(),
    ],
)
evaluator.run()
```

After running we can access the results in the `evaluator.results` attribute.
Alternatively, we can save the results to a file using `evaluator.save("results.jsonl")`.

The results are a list of dictionaries, where each dictionary contains the results for a single example.

We can also access the aggregated results in the `evaluator.aggregated_results` attribute.
