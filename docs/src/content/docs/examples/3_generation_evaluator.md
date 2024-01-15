---
title: Generation Evaluator
---

In this example, we show how to run generation evaluation over an example dataset.

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import DeterministicAnswerCorrectness

# Let's download the retrieval dataset example
dataset = example_data_downloader("correctness")

# Setup the evaluator
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
        DeterministicAnswerCorrectness(),
    ],
)

# Run the eval!
evaluator.run(batch_size=1)  # If using sematic metrics with GPU a bigger batch size is recommended

# Peaking at the results
print(evaluator.aggregated_results)

# Saving the results for future use
evaluator.save("generation_evaluator_results.jsonl")
```