---
title: Retrieval Evaluator
---

```python
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.dataset import Dataset
from continuous_eval.metrics import (
  PrecisionRecallF1,
  RankedRetrievalMetrics,
  MatchingStrategy,
)

dataset = Dataset.from_jsonl("data/retrieval.jsonl")
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
      PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
      RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
    ],
)
evaluator.run(k=2)
```

After running we can access the results in the `evaluator.results` attribute.
Alternatively, we can save the results to a file using `evaluator.save("results.jsonl")`.

The results are a list of dictionaries, where each dictionary contains the results for a single example.

We can also access the aggregated results in the `evaluator.aggregated_results` attribute.
