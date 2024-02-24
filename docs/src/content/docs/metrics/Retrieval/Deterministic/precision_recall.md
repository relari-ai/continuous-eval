---
title: Context Precision & Recall
---

### Definitions

**Context Precision: measures signal vs. noise** — what proportion of the retrieved contexts are relevant?

$$
\text{Context Precision} = \frac{\text{Relevant Retrieved Contexts}}{\text{All Retrieved Contexts}}
$$
<br>

**Context Recall: measures completeness** — what proportion of all relevant contexts are retrieved?

$$
\text{Context Recall} = \frac{\text{Relevant Retrieved Contexts}}{\text{All Ground Truth Contexts}}
$$

<br>

**F1:** harmonic mean of precision and recall

$$
\text{F1 Score} = 2 \times \frac{\text{Context Precision} \times \text{Context Recall}}{\text{Context Precision} + \text{Context Recall}}
$$


:::tip

**Context Recall should be the North Star metric for retrieval.**
This is because a retrieval system is only acceptable for generation if there is confidence that the retrieved context is complete enough to answer the question

:::


##### Matching Strategy

Given that the ground truth contexts can be defined differently from the exact chunks retrieved. For example, a ground truth contexts can be a sentence that contains the information, while the contexts retrieved are uniform 512-token chunks. We have following matching strategies that determine relevance:

<style>
    code {
        white-space: nowrap;
    }
</style>

<div style="overflow-x:auto; font-size: small">
    <table cellpadding="5" cellspacing="0">
        <thead>
            <tr>
                <th>Match Type</th>
                <th>Component</th>
                <th>Retrieved Component Considered relevant if:</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>ExactChunkMatch()</code></td>
                <td>Chunk</td>
                <td>Exact match to a Ground Truth Context Chunk.</td>
            </tr>
            <tr>
                <td><code>ExactSentenceMatch()</code></td>
                <td>Sentence</td>
                <td>Exact match to a Ground Truth Context Sentence.</td>
            </tr>
            <tr>
                <td><code>RoughChunkMatch()</code></td>
                <td>Chunk</td>
                <td>Match to a Ground Truth Context Chunk with ROUGE-L Recall &gt; <code>ROUGE_CHUNK_MATCH_THRESHOLD</code> (default 0.7).</td>
            </tr>
            <tr>
                <td><code>RougeSentenceMatch()</code></td>
                <td>Sentence</td>
                <td>Match to a Ground Truth Context Sentence with ROUGE-L Recall &gt; <code>ROUGE_CHUNK_SENTENCE_THRESHOLD</code> (default 0.8).</td>
            </tr>
        </tbody>
    </table>
</div>

### Example Usage

Required data items: `retrieved_context`, `ground_truth_context`

```python
from continuous_eval.metrics import PrecisionRecallF1, RougeChunkMatch

datum = {
    "retrieved_context": [
        "Paris is the capital of France and also the largest city in the country.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_context": ["Paris is the capital of France."],
}

metric = PrecisionRecallF1(RougeChunkMatch())
print(metric(**datum))
```

### Example Output

```JSON
{
    'context_precision': 0.5, 
    'context_recall': 1.0, 
    'context_f1': 0.6666666666666666
}
```

:::note

**You can run Precision / Recall / F1 @ top K** to see retrieval performance at the top K chunks over a dataset.
Check out <a href="/evaluators/evaluator#retrievalevaluator">Retrieval Evaluator</a> for examples
:::