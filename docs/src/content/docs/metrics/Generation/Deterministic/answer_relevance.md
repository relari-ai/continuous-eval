---
title: Answer Correctness
---
### Definitions

Answer Correctness measures how close the generated answer is the the ground truth reference answers.

Below are the list of deterministic metrics that measure the relationship between the generated answer and the ground truth reference answers.

**ROUGE-L** measures the longest common subsequence between the generated answer and the ground truth answers.

<br>

**Token Overlap** calculates the token overlap between the generated answer and the ground truth answers.

<br>


**BLEU** (Bilingual Evaluation Understudy) calculates the n-gram precision. (Below: `p_n` is the n-gram precision, `w_n` is the weight for each n-gram, and `BP` is the brevity penalty to penalize short answers)

$$
\text{BLEU Score} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$


<br>


**Answer Correctness** is a basket of metrics that include the **Precision, Recall and F1** of **ROUGE-L** and **Token Overlap**, as well as the **BLEU** score.

When there are multiple ground truth reference answers, the max score is taken.

<br>


:::note
**Token Overlap Recall and Rouge L Recall are shown to the best metrics in our experiment.** [Correntness Metric Evaluation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d) (scroll to "result for Correctness").
However, this conclusion likely varies by dataset. Test to see how close these scores align with human evaluation.
:::


### Example Usage

```python
from continuous_eval.metrics import DeterministicAnswerCorrectness

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

metric = DeterministicAnswerCorrectness()
print(metric.calculate(**datum))
```

### Example Output

```JSON
{
    'rouge_l_recall': 1.0, 
    'rouge_l_precision': 0.8, 
    'rouge_l_f1': 0.7272727223140496, 
    'token_overlap_recall': 1.0, 
    'token_overlap_precision': 0.8333333333333334, 
    'token_overlap_f1': 0.8333333333333334, 
    'bleu_score': 0.799402901304756
}
```
