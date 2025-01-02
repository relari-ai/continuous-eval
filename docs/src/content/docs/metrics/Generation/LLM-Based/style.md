---
title: LLM-based Style Consistency
---

### Definition

**LLM-based Style Consistency** outputs a score between 0.0 - 1.0 assessing the relevance and completeness of the generated answer based on the question. It assess style aspects such as tone, verbosity, formality, complexity, use of terminology, etc.

**Scoring rubric in LLM Prompt:**

- 0.0 means that the answer is in a completely different style as the reference answer(s).
- 0.33 means that the answer is barely in the same style as the reference answer(s), with noticable differences.
- 0.66 means that the answer is largely in the same style as the reference answer(s) but there's a slight difference in some aspects.
- 1.0 means that there's no dicernable style difference between the generated answer and reference answer(s).

### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics.generation.text import StyleConsistency

datum = {
    "answer": "Quantum computers work by utilizing quantum mechanics principles, specifically using qubits for complex computations.",
    "ground_truth_answers": [
        "A quantum computer is like having a super magical brain that can think about lots of different things all at the same time, really fast!"
    ]
}

metric = StyleConsistency()
print(metric(**datum))
```

### Sample Output

```python
{
    "consistency": 0.2500002355611551,
    "reasoning": "The generated answer is highly technical and formal. The reference answer uses a child-friendly comparison, making it much more accessible, while the generated answer would likely be less understandable to a general audience. This demonstrates a significant difference in style.",
}
```
