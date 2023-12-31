---
title: Getting started
description: How to install continuous-eval
---

## Installation

```bash
python3 -m pip install continuous-eval
```

## Run a metric

```bash
from continuous_eval.metrics import PrecisionRecallF1

datum = {
    "question": "Did Fargo win the golden globe nominations for both seasons?",
    "retrieved_contexts": [
        "Fargo is an American black comedy crime drama television series created and primarily written by Noah Hawley. The show is inspired by the 1996 film of the same name, which was written and directed by the Coen brothers, and takes place within the same fictional universe.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "ground_truth_contexts": [
        "The first season, set primarily in Minnesota and North Dakota from January 2006 to February 2007 and starring Billy Bob Thornton, Allison Tolman, Colin Hanks, and Martin Freeman, received wide acclaim from critics. It won the Golden Globe Awards for Best Miniseries or Television Film and Best Actor – Miniseries or Television Film for Thornton.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "answer": "Yes they did.",
    "ground_truths": [
        "Yes, they did."
        "Yes, Fargo received Golden Globe nominations in season 1 and 2.",
        "Yes, Fargo received three nominations for season 1 and three nominations in season 2."
    ],
}

metric = PrecisionRecallF1()
print(metric.calculate(**datum))
```

## Run eval on a dataset

**Load Golden Dataset**

```python
from continuous_eval.dataset import Dataset
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import (
  PrecisionRecallF1,
  RankedRetrievalMetrics,
  MatchingStrategy,
)

# Load golden dataset
dataset = Dataset.from_jsonl("data/example_dataset.jsonl")

# Select Retrieval or Generation Evaluator
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
      PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
      RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
    ],
)
evaluator.run(k=2)
```

If you don't have a golden dataset, you can use `SimpleDatasetGenerator` to create a silver dataset.
```python
from continuous_eval.simple_dataset_generator import SimpleDatasetGenerator

dataset = SimpleDatasetGenerator(VectorStoreIndex, num_questions=10)
```

#### Read results