---
title: Token Count
---

### Definitions

Token Count calculates the number of tokens used in the retrieved context.

A required input for the metrics is `encoder_name` for tiktoken. 

For example, for the most recent OpenAI models, you use `cl100k_base` as the encoder. For other models, you should look up the specific tokenizer used, or alternatively, you can also use `approx` to get an approximate token count which measures 1 token for every 4 characters.

:::tip
**Tokens in `retrieved_context` often accounts for the majority of LLM token usage in a RAG application.** 
Token count is useful to keep track of if you are concerned about LLM cost, LLM context window limit, and LLM performance issued caused by low context precision (such as "needle-in-a-haystack" problems).
:::

Required data items: `retrieved_context`

```python
from continuous_eval.metrics.retrieval import TokenCount

datum = {
    "retrieved_context": [
        "Lyon is a major city in France.",
        "Paris is the capital of France and also the largest city in the country.",
    ],
    "ground_truth_context": ["Paris is the capital of France."],
}

metric = TokenCount(encoder_name="cl100k_base")
print(metric(**datum))
```

### Example Output

```JSON
{
    'num_tokens': 24, 
}
```
