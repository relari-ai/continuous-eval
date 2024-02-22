---
title: BERT Answer Similarity
---

### Definitions

**BERT Answer Similarity** measures the semantic similarity between the Generated Answer and the Ground Truth Answers.

This metric leverages the [BERT model](https://huggingface.co/bert-base-uncased) to calculate semantic similarity.

<br>

:::note
Semantic similarity between `answer` and `ground_truths` is not necessarily a good indication that the answer is correct. Test the metric to see how well correlated with human judgement.
:::

### Example Usage

Required data items: `answer`, `ground_truths`

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

The metric outputs the max BERT similarity score calculated using items in `ground_truths`

```JSON
{
    'bert_answer_similarity': 0.9274404048919678
}
```
