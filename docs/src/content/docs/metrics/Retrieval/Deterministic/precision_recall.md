---
title: Precision/Recall/F1
---

### Definition
#### Precision
\[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \]

#### Recall
\[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \]

#### F1
\[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

#### Matching Strategy
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
                <th><code>Component</code></th>
                <th>Retrieved <code>Component</code> Considered relevant if:</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>EXACT_CHUNK_MATCH</code></td>
                <td>Chunk</td>
                <td>Exact match to a Ground Truth Context Chunk.</td>
            </tr>
            <tr>
                <td><code>EXACT_SENTENCE_MATCH</code></td>
                <td>Sentence</td>
                <td>Exact match to a Ground Truth Context Sentence.</td>
            </tr>
            <tr>
                <td><code>ROUGE_CHUNK_MATCH</code></td>
                <td>Chunk</td>
                <td>Match to a Ground Truth Context Chunk with ROUGE-L Recall &gt; <code>ROUGE_CHUNK_MATCH_THRESHOLD</code> (default 0.7).</td>
            </tr>
            <tr>
                <td><code>ROUGE_SENTENCE_MATCH</code></td>
                <td>Sentence</td>
                <td>Match to a Ground Truth Context Sentence with ROUGE-L Recall &gt; <code>ROUGE_CHUNK_SENTENCE_THRESHOLD</code> (default 0.8).</td>
            </tr>
        </tbody>
    </table>
</div>


### Example Usage
```python
from continuous_eval.metrics import PrecisionRecallF1, MatchingStrategy

metric = PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH)
```
### Example Output
```JSON
{
    "precision": 0.5,
    "recall": 1.0,
    "f1": 0.877777777
}
```