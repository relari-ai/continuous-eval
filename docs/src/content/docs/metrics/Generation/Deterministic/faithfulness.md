---
title: Faithfulness
sidebar:
    order: 2
---

### Definitions

**Faithfulness** measures how grounded is the generated answer on the retrieved contexts. 

Below are the list of deterministic metrics that measure the relationship between the generated answer and the retrieved contexts.

**ROUGE-L Precision** measures the longest common subsequence between the generated answer and the retrieved contexts.

$$
\text{ROUGE-L Precision} = \frac{\text{Longest Common Subsequence (LCS) between Answer and Contexts}}{\text{Length of Generated Answer}}
$$

<br>

**Token Overlap Precision** calculates the precision of token overlap between the generated answer and the retrieved contexts.

$$
\text{Token Overlap Precision} = \frac{\text{Count of Common Tokens between Answer and Contexts}}{\text{Total Tokens in Generated Answer}}
$$

<br>


**BLEU** (Bilingual Evaluation Understudy) calculates the n-gram precision. (Below: `p_n` is the n-gram precision, `w_n` is the weight for each n-gram, and `BP` is the brevity penalty to penalize short answers)

$$
\text{BLEU Score} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$


<br>


**Rouge|Token Overlap|Bleu Faithfulness** is defined as the proportion of the sentences in the generated answer that can matched to the retrieved context above a threshold.


$$
\text{Rouge|Token Overlap|Bleu Faithfulness} = \frac{\text{Number of Sentences in Answer Matched to Context above Threshold}}{\text{Total Number of Sentences in Generated Answer}}
$$

<br>


### Example Usage

Required data items: `retrieved_contexts`, `answer`

```python
from continuous_eval.metrics import DeterministicFaithfulness

datum = {
    "retrieved_contexts": ["William Shakespeare is the author of 'Romeo and Juliet'."],
    "answer": "William Shakespeare wrote 'Romeo and Juliet'. He is born in Ireland",
}

metric = DeterministicFaithfulness()
print(metric(**datum))
```

### Example Output

`by_sentence` values are the list of sentence-level rouge | token_overlap | bleu scores for each sentence in the answer.

`default_threshold` for a sentence to be considered faithful is set to be `0.5`.

```JSON
{
    'rouge_faithfulness': 0.5, 
    'token_overlap_faithfulness': 0.5, 
    'bleu_faithfulness': 0.37023896751607194, 
    'rouge_p_by_sentence': [0.8333333333333334, 0.2], 
    'token_overlap_p_by_sentence': [0.875, 0.2], 
    'bleu_score_by_sentence': [0.6855956729300113, 0.05488226210213251]
}
```
