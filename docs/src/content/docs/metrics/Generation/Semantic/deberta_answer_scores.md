---
title: DeBERTa Answer Scores
---

### Definitions

**DeBERTa scores** measure semantic relationship between the Generated Answer and the Ground Truth Answers **in three categories:**

- **Entailment**: the Generated Answer IMPLIES a Ground Truth Answer.

- **Contradiction**: the Generated Answer CONTRADICTS a Ground Truth Answer.

- **Neutral**: the Generated Answer and the Ground Truth Answer have neutral logical relationship.

This metric leverages the [NLI DeBERTa v3 model](https://huggingface.co/cross-encoder/nli-deberta-v3-large) to calculate the scores. This DeBERTa model (Decoding-enhanced BERT with Disentangled Attention) is a fine-tuned version specifically designed to measure the above relationships.

**The scores output the probability of the model's prediction of each class (between 0 and 1).** Because we are mostly interested in finding out if entailment or contradiction relationships, our scores only output those two.

<br>

:::tip
**The DeBERTa scores take it one step further than the BERT metrics which only measures semantic closeness.** In the context of RAG, it is a more nuanced assessment of answer quality, correlates better with human evaluation.
[Correntness Metric Evaluation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d) (scroll to "result for Correctness").
:::


### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics import DebertaAnswerScores

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

metric = DebertaAnswerScores()
print(metric.calculate(**datum))

reverse_metric = DebertaAnswerScores(reverse=True)
print(reverse_metric.calculate(**datum))
```

### Example Output

Default: `reverse = False`

```JSON
{
    'deberta_answer_entailment': 0.9989350438117981, 
    'deberta_answer_contradiction': 2.3176469767349772e-05
}
```

The above scores suggests that the model is highly confident that the Generate Answer implies at least one of the Ground Truth Answers, and that it unlikely contradicts with any of them.

Default: `reverse = True`

```JSON
{
    'deberta_reverse_answer_entailment': 0.9990990161895752, 
    'deberta_reverse_answer_contradiction': 3.902518074028194e-05
}
```

The above scores suggests that the model is highly confident that the Generate Answer implies at least one of the Ground Truth Answers, and that it unlikely contradicts with any of them.