---
title: Evaluation Dataset Preparation
---

In this example, we show how to prepare, load and save the dataset for evaluation.

```python
from continuous_eval import Dataset

# A dataset is a list of dictionaries, each representing, retrieved context etc...
# Common fields are:
# - "question"  # user question
# - "retrieved_contexts"
# - "ground_truth_contexts"
# - "answer"  # generated answer
# - "ground_truths"  # ground truth answers

q_1 = {
    "question": "What is the capital of France?",
    "retrieved_contexts": [
        "Paris is the largest city in France.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_contexts": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}
q_2 = {
    "question": "What's the Ultimate Question of Life, the Universe, and Everything?",
    "retrieved_contexts": [
        "Energy equals mass times the speed of light squared.",
    ],
    "ground_truth_contexts": ["42"],
    "answer": "I don't know",
    "ground_truths": ["42"],
}
dataset = Dataset([q_1, q_2])
print(f"The size of the dataset is: {len(dataset)}")

# We can iterate over the dataset
for datum in dataset.iterate():
    print(datum)

# Or simply get a datum
print(f"The first question is: {dataset.datum(0)['question']}")

# The dataset extends a Pandas DataFrame, so we can also use be used as such.
num_idk = dataset['answer'].str.contains("I don't know", case=False, na=False).sum()
print(f"There {'are' if num_idk > 1 else 'is'} {num_idk} 'I don't know' answers in the dataset")

# # We can export (save) the dataset to a jsonl file
dataset.to_jsonl("my_dataset.jsonl")

# # We can also load a dataset from a jsonl file
dataset = Dataset.from_jsonl("my_dataset.jsonl")
```