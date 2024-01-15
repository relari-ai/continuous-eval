---
title: LLM-based Faithfulness
---

### Definition

**LLM-based Faithfulness** measures how grounded is the generated answer on the retrieved contexts. 

We have two different ways of prompting the LLM to calculate faithfulness:

**Classify faithfulness by statement**:


`classify_by_statement = TRUE` where LLM is prompted to evaluate the faithfulness of each statement in the Generated Answer and outputs a `float` score:

$$
\text{LLM-Based Faithfulness} =
\frac{
  \text{Number of Statements in Generated Answer Attributed to the Retrieved Contexts}
}{
  \text{Total Number of Statements in Generated Answer}
}
$$

<br>


**Classify faithfulness by whole answer**:


`classify_by_statement = FALSE` where LLM is prompted to evaluate the whole Generated Answer and outputs a `boolean` judgement

$$
\text{LLM-Based Faithfulness} =
{
  \text{The Generated Answer is fully based on Retrieved Contexts}
}
$$

<br>

:::note
**Classify by statement provides a more granular view and is generally more reliable.** Because it asks the LLM to reason over each statement. However it requires more tokens than classifying by whole answers.
:::


### Example Usage

Required data items: `question`, `retrieved_context`, `answer`

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


`faithfulness_by_statement` output:
```JSON
{
    'LLM_based_faithfulness': 0.5, 
    'LLM_based_faithfulness_reasoning': '{
        "classification": [
            {
                "statement_1": "William Shakespeare wrote \'Romeo and Juliet\'.",
                "reason": "This is directly stated in the context.",
                "Attributed": 1
            },
            {
                "statement_2": "He is born in Ireland.",
                "reason": "The context does not provide information about his birthplace, and the statement is factually incorrect as William Shakespeare was born in England.",
                "Attributed": 0
            }
        ]
    }
}
```

`faithfulness_by_whole_answer` output:
```JSON
{
    'LLM_based_faithfulness': False, 
    'LLM_based_faithfulness_reasoning': "The statement that William Shakespeare wrote 'Romeo and Juliet' is supported by the context. However, the context does not provide information about his birthplace, and it is a well-known fact that William Shakespeare was born in Stratford-upon-Avon, England, not Ireland."
}
```
