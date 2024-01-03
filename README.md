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
poetry install --all-extras
```

## Getting Started

### Prerequisites

The code requires the `OPENAI_API_KEY` (optionally `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY`) in .env to run the LLM-based metrics.

### Usage

```python
from continuous_eval.metrics import PrecisionRecallF1, RougeChunkMatch

datum = {
    "question": "What is the capital of France?",
    "retrieved_contexts": [
        "Paris is the capital of France and its largest city.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_contexts": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}

metric = PrecisionRecallF1(RougeChunkMatch())
print(metric.calculate(**datum))
```

To run over a dataset, you can use one of the evaluator classes:

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics

# Build a dataset: create a dataset from a list of dictionaries containing question/answer/context/etc.
# Or download one of the of the examples... 
dataset = example_data_downloader("retrieval")
# Setup the evaluator
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
        PrecisionRecallF1(),
        RankedRetrievalMetrics(),
    ],
)
# Run the eval!
evaluator.run(k=2, batch_size=1)
# Peaking at the results
print(evaluator.aggregated_results)
# Saving the results for future use
evaluator.save("retrieval_evaluator_results.jsonl")
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

- `DeterministicAnswerCorrectness`: Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- `DeterministicFaithfulness`: Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision and BLEU score

#### Semantic

- `DebertaAnswerScores`: Entailment and contradiction scores between the Generated Answer and Ground Truth Answer
- `BertAnswerRelevance`: Similarity score based on the BERT model between the Generated Answer and Question
- `BertAnswerSimilarity`: Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer

#### LLM-based

- `LLMBasedFaithfulness`: Binary classifications of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts by LLM
- `LLMBasedAnswerCorrectness`: Score (1-5) of the Generated Answer based on the Question and Ground Truth Answer calcualted by LLM

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.
