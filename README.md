# Continuous Evaluation for retrieval-based LLMs pipelines

Continuous evaluation for retrieval-based LLMs pipelines.

## Installation

This code is provided as a Python package. To install it, run the following command:

```bash
python3 -m pip install continuous-eval
```

if you want to install from source

```bash
git clone https://github.com/relari-ai/continuous-eval.git && cd continuous-eval
poetry install
```

## Getting Started

### Prerequisites

The code requires the `OPENAI_API_KEY` (and optionally an `ANTHROPIC_API_KEY`) to run the LLM-based metrics.

### Usage

```python
from continuous_eval.metrics import PrecisionRecallF1, MatchingStrategy

datum = {
    "question": "Did Fargo win the golden globe nominations for both seasons?",
    "retrieved_contexts": [
        "Fargo is an American black comedy crime drama television series created and primarily written by Noah Hawley. The show is inspired by the 1996 film of the same name, which was written and directed by the Coen brothers, and takes place within the same fictional universe. The Coens were impressed by Hawley's script and agreed to be named as executive producers.[3] The series premiered on April 15, 2014, on FX,[3] and follows an anthology format, with each season set in a different era and location, with a different story and mostly new characters and cast, although there is minor overlap. Each season is heavily influenced by various Coen brothers films, with each containing numerous references to them.[4]",
        "The first season, set primarily in Minnesota and North Dakota from January 2006 to February 2007 and starring Billy Bob Thornton, Allison Tolman, Colin Hanks, and Martin Freeman, received wide acclaim from critics.[5] It won the Primetime Emmy Awards for Outstanding Miniseries, Outstanding Directing, and Outstanding Casting, and received 15 additional nominations including Outstanding Writing, another Outstanding Directing nomination, and acting nominations for all four leads. It also won the Golden Globe Awards for Best Miniseries or Television Film and Best Actor – Miniseries or Television Film for Thornton.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "ground_truth_contexts": [
        "The first season, set primarily in Minnesota and North Dakota from January 2006 to February 2007 and starring Billy Bob Thornton, Allison Tolman, Colin Hanks, and Martin Freeman, received wide acclaim from critics.[5] It won the Primetime Emmy Awards for Outstanding Miniseries, Outstanding Directing, and Outstanding Casting, and received 15 additional nominations including Outstanding Writing, another Outstanding Directing nomination, and acting nominations for all four leads. It also won the Golden Globe Awards for Best Miniseries or Television Film and Best Actor – Miniseries or Television Film for Thornton.",
        "The second season, set in Minnesota, North Dakota, and South Dakota in March 1979 and starring Kirsten Dunst, Patrick Wilson, Jesse Plemons, Jean Smart, Allison Tolman, and Ted Danson, received widespread critical acclaim.[6] It received three Golden Globe nominations, along with several Emmy nominations including Outstanding Miniseries, and acting nominations for Dunst, Plemons, Smart, and Bokeem Woodbine.",
    ],
    "answer": "Berlin",
    "ground_truths": [
        "Yes, they did get a nomination in season 1 and 2.",
        "Not really, they didn't win for season three.",
    ],
}

metric = PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH)
print(metric.calculate(**datum))
```

To run over a dataset, you can use one of the evaluator classes:

```python
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import (
    MatchingStrategy,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


dataset = ... # load your dataset as list of dictionaries
evaluator = RetrievalEvaluator(
  [
    PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
    RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
  ]
)
results = evaluator.run(dataset, aggregate=True)
```

For generation you can instead use the `GenerationEvaluator`.

## Metrics

### Retrieval-based metrics

#### Deterministic

- `PrecisionRecallF1`: Rank-agnostic metrics including Precision, Recall, and F1 of Retrieved Contexts
- `RankedRetrievalMetrics`: Rank-aware metrics including Mean Average Precision (MAP), Mean Reciprical Rank (MRR), NDCG (Normalized Discounted Cumulative Gain) of retrieved contexts

#### LLM-based

- `LLMBasedContextPrecision`: Precision and Mean Average Precision (MAP) based on context relevancy classified by LLM
- `LLMBasedContextCoverage`: Proportion of statements in ground truth answer that can be attributed to Retrieved Contexts calcualted by LLM

### Generation-based metrics

#### Deterministic

- `DeterministicAnswerRelevance`: Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- `RougeSentenceFaithfulness`: Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L recall
- `BertAnswerRelevance`: Similarity score based on the BERT model between the Generated Answer and Question
- `BertAnswerSimilarity`: Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer

#### LLM-based

- `LLMBasedFaithfulness`: Classification of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts by LLM
- `LLMBasedAnswerCorrectness`: Score (1-5) of the Generated Answer based on the Question and Ground Truth Answer calcualted by LLM

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.
