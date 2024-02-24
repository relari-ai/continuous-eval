---
title: LLM-based Correctness
---

### Definition


**LLM-based Answer Correctness** outputs a score between 0.0 - 1.0 assessing the overall quality of the answer, given the question and ground truth answer. 

**Scoring rubric in LLM Prompt:**
- 0.0 means that the answer is completely irrelevant to the question.
- 0.25 means that the answer is relevant to the question but contains major errors.
- 0.5 means that the answer is relevant to the question and is partially correct.
- 0.75 means that the answer is relevant to the question and is correct.
- 1.0 means that the answer is relevant to the question and is correct and complete.



### Example Usage

Required data items: `question`, `answer`, `ground_truths`

```python
from continuous_eval.metrics import LLMBasedAnswerCorrectness
from continuous_eval.llm_factory import LLMFactory

datum = {
    "question": "Who wrote 'Romeo and Juliet'?",
    "answer": "Shakespeare wrote 'Romeo and Juliet'",
    "ground_truths": [
        "William Shakespeare wrote 'Romeo and Juliet", 
        "William Shakespeare", 
        "Shakespeare", 
        "Shakespeare is the author of 'Romeo and Juliet'"
    ]
}

metric = LLMBasedAnswerCorrectness(LLMFactory("gpt-4-1106-preview"))
print(metric(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_answer_correctness': 1.0, 
    'LLM_based_answer_correctness_reasoning': 'The answer is relevant to the question and is correct and complete. It matches the ground truth reference answers provided.'
}
```
