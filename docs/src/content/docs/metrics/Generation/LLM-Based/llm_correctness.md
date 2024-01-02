---
title: LLM-based Correctness
---

### Definition


You need to evaluate the quality of the generated answer based on the question and reference ground truth answer.
Output a score and the reasoning for your score in a new line.
Use the following guidelines for evaluation:
* You should output a single score between 1 to 5.
* 1 means that the answer is completely irrelevant to the question.
* 2 means that the answer is relevant to the question but contains major errors.
* 3 means that the answer is relevant to the question and is partially correct.
* 4 means that the answer is relevant to the question and is correct.
* 5 means that the answer is relevant to the question and is correct and complete.

<br>



### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics import LLMBasedAnswerCorrectness
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

metric = LLMBasedAnswerCorrectness(LLMFactory("gpt-4-1106-preview"))
print(metric.calculate(**datum))
```

### Sample Output

```JSON
{
    'LLM_based_answer_correctness': 5.0, 
    'LLM_based_answer_correctness_reasoning': 'The answer is relevant to the question and is correct and complete. It matches the ground truth reference answers provided.'}
```
