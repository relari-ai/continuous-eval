---
title: Evaluator Class
---

## `Evaluator` Class 

The `Evaluator` takes a `Dataset` and a list of `Metric` and outputs the evaluation results.

Use `RetrievalEvaluator` for Retrieval Metrics and `GenerationEvaluator` for Generation Metrics.

### `RetrievalEvaluator`

#### Example Usage

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics

# Build a dataset: create a dataset from a list of dictionaries containing question/answer/context/etc.
# Or download one of the of the examples... 
dataset = example_data_downloader("retrieval")
# Setup the evaluator
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
        PrecisionRecallF1(),
        RankedRetrievalMetrics(),
    ],
)
# Run the eval!
evaluator.run(k=2, batch_size=1)
# Peak at the results
print(evaluator.aggregated_results)
# Saving the results for future use
evaluator.save("retrieval_evaluator_results.jsonl")
```
:::tip
**Set k in evaluators.run() variable to run retrieval metrics @ top K chunks in the dataset.** Understanding precision, recall and other metrics at varying K chunks can help you assess how well your system performs at various top K. 

**Check out ["Metric @ K"](https://medium.com/relari/a-practical-guide-to-rag-pipeline-evaluation-part-1-27a472b09893)** in A Practical Guide to RAG Pipeline Evaluation (scroll to "Action 2: Use [metric]@K") for a deeper dive.
:::

#### Sample Output

```bash
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:00<00:00, 29039.05it/s]
{'average_precision': 0.575, 'reciprocal_rank': 0.575, 'ndcg': 0.4696839251404123, 'context_precision': 0.4666666666666667, 'context_recall': 0.4666666666666667, 'context_f1': 0.4666666666666667}
```


### `GenerationEvaluator`

#### Example Usage

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import GenerationEvaluator
from continuous_eval.metrics import DeterministicAnswerCorrectness, DebertaAnswerScores

dataset = example_data_downloader("correctness")[:100] # Run over the first 100 datapoints
evaluator = GenerationEvaluator(
    dataset=dataset,
    metrics=[
        DeterministicAnswerCorrectness(),
        DebertaAnswerScores(),
    ],
)

evaluator.run(batch_size=20)
print(evaluator.aggregated_results)
evaluator.save("generation_evaluator_results.jsonl")
```

:::tip
Semantic metrics such as `DebertaAnswerScores` can be more efficiently calculated in batches.
:::

#### Sample Output

```bash
Processing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:43<00:00,  2.28it/s]
{'deberta_answer_entailment': 1.0455470567569136, 'deberta_answer_contradiction': -2.7038792559504508, 'rouge_l_recall': 0.7281666666666666, 'rouge_l_precision': 0.11723592829981358, 'rouge_l_f1': 0.1892981846941943, 'token_overlap_recall': 0.835, 'token_overlap_precision': 0.11119536790082339, 'token_overlap_f1': 0.18378446760182654, 'bleu_score': 0.12819215763848693}
```