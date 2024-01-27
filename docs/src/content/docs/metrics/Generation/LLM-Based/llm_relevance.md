---
title: LLM-based Answer Relevance
---

### Definition


**LLM-based Answer Relevance** outputs a score between 1 - 3 assessing the relevance and completeness of the generated answer based on the question.


**Scoring rubric in LLM Prompt:**
- 1 means that the answer is completely irrelevant to the question.
- 2 means that the answer is partially relevant to the question or it only partially answers the question.
- 3 means that the answer is relevant to the question and completely answers the question.



### Example Usage

Required data items: `question`, `answer`

```python
from continuous_eval.metrics import LLMBasedAnswerRelevance
from continuous_eval.llm_factory import LLMFactory

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

metric = LLMBasedAnswerRelevance(LLMFactory("gpt-4-1106-preview"))
print(metric.calculate(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_answer_relevance': 3.0, 
    'LLM_based_answer_relevance_reasoning': "The answer is relevant to the question and completely answers the question by correctly identifying Shakespeare as the author of 'Romeo and Juliet'."
}
```
