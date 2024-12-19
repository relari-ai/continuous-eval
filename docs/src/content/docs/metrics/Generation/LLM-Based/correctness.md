---
title: AnswerCorrectness
---

### Definition

** Answer Correctness** outputs a score between 0.0 - 1.0 assessing the overall quality of the answer, given the question and ground truth answer. 

**Scoring rubric in LLM Prompt:**

- 0.0 means that the answer is completely irrelevant to the question.
- 0.25 means that the answer is relevant to the question but contains major errors.
- 0.5 means that the answer is relevant to the question and is partially correct.
- 0.75 means that the answer is relevant to the question and is correct.
- 1.0 means that the answer is relevant to the question and is correct and complete.

### Example Usage

Required data items: `question`, `answer`, `ground_truths`

```python
from continuous_eval.metrics.generation.text import AnswerCorrectness

datum = {
    "question": "Who wrote 'Romeo and Juliet'?",
    "answer": "Shakespeare wrote 'Romeo and Juliet'",
    "ground_truth_answers": [
        "William Shakespeare wrote 'Romeo and Juliet", 
        "William Shakespeare", 
        "Shakespeare", 
        "Shakespeare is the author of 'Romeo and Juliet'"
    ]
}

metric = AnswerCorrectness()
print(metric(**datum))
```

### Sample Output

```python
{
    "correctness": 0.9999867895679586,
    "reasoning": "The generated answer correctly identifies Shakespeare as the author of 'Romeo and Juliet'.",
}
```
