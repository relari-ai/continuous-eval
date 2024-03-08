---
title: Ranked-Aware Metrics
---

### Definitions

Rank-aware metrics takes into account the order in which the contexts are retrieved.

**Average Precision (AP)** measures all relevant chunks retrieved and calculates weighted score. Mean of AP across dataset is frequently referred to as **MAP**.

$$ \text{Average Precision (AP)} = \frac{1}{\text{Number of Relevant Documents}} \sum_{j=1}^{\text{Number of Retrieved Documents}} \text{Precision at Rank } j $$

<br>

**Reciprocal Rank (RR)** measures when your **first relevant chunk** appear in your retrieval. Mean of RR across dataset is frequently referred to as **MRR**.

$$ \text{Reciprocal Rank (RR)} = \frac{1}{\text{Rank of First Relevant Document}} $$

<br>

**Normalized Discounted Cumulative Gain (NDCG)** accounts for the cases where your classification of relevancy is non-binary. 

$$ \text{Normalized Discounted Cumulative Gain (NDCG)} = \frac{\text{DCG at Rank } k}{\text{IDCG at Rank } k} $$

<br>


:::tip
Focus on **MRR if a single chunk typically contains all the information** needed to answer a question.
Focus on **MAP if multiple chunks need to be synthesized** to answer a question.
:::

##### Matching Strategy

Please checkout explanation for Matching strategy in [Matching Strategy](/../precision_recall/)

### Example Usage

Required data items: `retrieved_context`, `ground_truth_context`

```python
from continuous_eval.metrics.retrieval import RankedRetrievalMetrics, RougeChunkMatch

datum = {
    "retrieved_context": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truth_context": ["Paris is the capital of France."],
}

metric = RankedRetrievalMetrics(RougeChunkMatch())
print(metric(**datum))
```

### Example Output

```JSON
{
    'average_precision': 0.5, 
    'reciprocal_rank': 0.5, 
    'ndcg': 0.6309297535714574
}
```
