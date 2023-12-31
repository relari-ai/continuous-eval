---
title: Retrieval metrics
description: Overview of different types of metrics
---

## Deterministic

### Precision, Recall, F1 (rank-agnostic metrics)
-   Precision
-   Recall
-   F1
```python
from continuous_eval.metrics import PrecisionRecallF1

metric = PrecisionRecallF1()
metric.calculate(**datum)
```

### Rank-aware metrics
-   Mean Average Precision (MAP)
-   Mean Reciprical Rank (MRR)
-   NDCG (Normalized Discounted Cumulative Gain)
```python
from continuous_eval.metrics import RankedRetrievalMetrics

metric = RankedRetrievalMetrics()
metric.calculate(**datum)
```


## LLM-based

- `LLMBasedContextPrecision`: Precision and Mean Average Precision (MAP) based on context relevancy classified by LLM
- `LLMBasedContextCoverage`: Proportion of statements in ground truth answer that can be attributed to Retrieved Contexts calcualted by LLM
