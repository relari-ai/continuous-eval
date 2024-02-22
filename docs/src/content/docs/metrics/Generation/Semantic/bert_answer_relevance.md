---
title: BERT Answer Relevance
---

### Definitions

**BERT Answer Relevance** measures the semantic similarity between the Generated Answer and the Question

This metric leverages the [BERT model](https://huggingface.co/bert-base-uncased) to calculate semantic similarity.

<br>

:::note
Semantic similarity between `answer` and `question` is not necessarily a good indication that the answer is relevant to the question. Test the metric to see how well correlated with human judgement.
:::

### Example Usage

Required data items: `question`, `answer`

```python
from continuous_eval.metrics import BertAnswerSimilarity

datum = {
    "question": "Who wrote 'Romeo and Juliet'?",
    "retrieved_contexts": ["William Shakespeare is the author of 'Romeo and Juliet'."],
    "ground_truth_contexts": ["William Shakespeare is the author of 'Romeo and Juliet'."],
    "answer": "Shakespeare wrote 'Romeo and Juliet'",
    "ground_truths": [
        "William Shakespeare wrote 'Romeo and Juliet", 
        "William Shakespeare", 
        "Shakespeare", 
        "Shakespeare is the author of 'Romeo and Juliet'"
    ]
}

metric = BertAnswerSimilarity()
print(metric(**datum))
```

### Example Output

```JSON
{
    'bert_answer_relevance': 0.8146507143974304
}
```
