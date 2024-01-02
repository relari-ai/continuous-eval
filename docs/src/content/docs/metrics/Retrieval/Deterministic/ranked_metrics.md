---
title: Ranked Metrics
---

### Definitions

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

Required data items: `retrieved_context`, `ground_truth_contexts`

```python
from continuous_eval.metrics import RankedRetrievalMetrics, RougeChunkMatch

datum = {
    "question": "What is the capital of France?",
    "retrieved_contexts": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truth_contexts": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}

metric = RankedRetrievalMetrics(RougeChunkMatch())
print(metric.calculate(**datum))
```

### Example Output

```JSON
{
    'Average Precision': 0.5, 
    'Reciprocal Rank': 0.5, 
    'NDCG': 0.6309297535714574
}
```
