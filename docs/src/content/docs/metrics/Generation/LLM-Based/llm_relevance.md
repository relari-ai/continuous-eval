---
title: LLM-based Answer Relevance
---

### Definition


**LLM-based Answer Relevance** outputs a score between 0.0 - 1.0 assessing the consistency of the generated answer based on the reference ground truth answers.


**Scoring rubric in LLM Prompt:**
- 0.0 means that the answer is completely irrelevant to the question.
- 0.5 means that the answer is partially relevant to the question or it only partially answers the question.
- 1.0 means that the answer is relevant to the question and completely answers the question.


### Example Usage

Required data items: `question`, `answer`

```python
from continuous_eval.metrics import LLMBasedAnswerRelevance
from continuous_eval.llm_factory import LLMFactory

datum = {
    "question": "Who wrote 'Romeo and Juliet'?",
    "answer": "Shakespeare wrote 'Romeo and Juliet'",
}

metric = LLMBasedAnswerRelevance(LLMFactory("gpt-4-1106-preview"))
print(metric(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_answer_relevance': 1.0, 
    'LLM_based_answer_relevance_reasoning': "The answer is relevant to the question and completely answers the question by correctly identifying Shakespeare as the author of 'Romeo and Juliet'."
}
```
