---
title: LLM-based Style Consistency
---

### Definition


**LLM-based Style Consistency** outputs a score between 0.0 - 1.0 assessing the relevance and completeness of the generated answer based on the question. It assess style aspects such as tone, verbosity, formality, complexity, use of terminology, etc.


**Scoring rubric in LLM Prompt:**
- 0.0 means that the answer is in a completely different style as the reference answer(s).
- 0.33333333333333333 means that the answer is barely in the same style as the reference answer(s), with noticable differences.
- 0.66666666666666666 means that the answer is largely in the same style as the reference answer(s) but there's a slight difference in some aspects.
- 1.0 means that there's no dicernable style difference between the generated answer and reference answer(s).



### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics import LLMBasedStyleConsistency
from continuous_eval.llm_factory import LLMFactory

datum = {
    "answer": "Quantum computers work by utilizing quantum mechanics principles, specifically using qubits for complex computations.",
    "ground_truths": [
        "A quantum computer is like having a super magical brain that can think about lots of different things all at the same time, really fast!"
    ]
}

metric = LLMBasedAnswerRelevance(LLMFactory("gpt-4-1106-preview"))
print(metric(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_style_consistency': 0.16666666666666666, 
    'LLM_based_style_consisntency_reasoning': 'The generated answer is formal, technical, and uses specific terminology like "quantum mechanics" and "qubits," whereas the reference answer uses a very informal, simplified, and metaphorical style to explain quantum computers.'
}
```
