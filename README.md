<h3 align="center">
  <img
    src="docs/public/continuous-eval-logo.png"
    width="350"
  >
</h3>

<div align="center">

  
  <a href="https://docs.relari.ai/" target="_blank"><img src="https://img.shields.io/badge/docs-view-blue" alt="Documentation"></a>
  <a href="https://pypi.python.org/pypi/continuous-eval">![https://pypi.python.org/pypi/continuous-eval/](https://img.shields.io/pypi/pyversions/continuous-eval.svg)</a>
  <a href="https://github.com/relari-ai/continuous-eval/releases">![https://GitHub.com/relari-ai/continuous-eval/releases](https://img.shields.io/github/release/relari-ai/continuous-eval)</a>
  <a href="https://pypi.python.org/pypi/continuous-eval/">![https://github.com/Naereen/badges/](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)</a>
  <a a href="https://github.com/relari-ai/continuous-eval/blob/main/LICENSE">![https://pypi.python.org/pypi/continuous-eval/](https://img.shields.io/pypi/l/continuous-eval.svg)</a>


</div>

<h2 align="center">
  <p>Open-Source Evaluation Framework for LLM Pipelines</p>
</h2>

## Overview

`continuous-eval` is an open-source package created for the scientific and practical evaluation of LLM application pipelines. Currently, it focuses on retrieval-augmented generation (RAG) pipelines.

## Why another eval package?

Good LLM evaluation should help developers reliably identify weaknesses in the pipeline, inform what actions to take, and accelerate development from prototype to production. Although it is optimal to put LLM Evaluation as part of our CI/CD pipeline just like any other part of software, it remains challenging today because:

**Human evaluation is trustworthy but not scalable**
- Eyeballing can only be done on a small dataset, and it has to be repeated for any pipeline update  
- User feedback is spotty and lacks granularity

**Using LLMs to evaluate LLMs is expensive, slow and difficult to trust**
- Can be very costly and slow to run at scale
- Can be biased towards certain answers and often doesn’t align well with human evaluation

## How is continuous-eval different?

- **Comprehensive RAG Metric Library**: mix and match Deterministic, Semantic and LLM-based metrics.

- **Trustworthy Ensemble Metrics**: easily build a close-to-human ensemble evaluation pipeline with mathematical guarantees.

- **Cheaper and Faster Evaluation**: our hybrid pipeline slashes cost by up to 15x compared to pure LLM-based metrics, and reduces eval time on large datasets from hours to minutes.

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

### Generation metrics

#### Deterministic

- `DeterministicAnswerCorrectness`: Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- `DeterministicFaithfulness`: Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision and BLEU score
- `FleschKincaidReadability`: how easy or difficult it is to understand the LLM generated answer.

#### Semantic

- `DebertaAnswerScores`: Entailment and contradiction scores between the Generated Answer and Ground Truth Answer
- `BertAnswerRelevance`: Similarity score based on the BERT model between the Generated Answer and Question
- `BertAnswerSimilarity`: Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer

#### LLM-based

- `LLMBasedFaithfulness`: Binary classifications of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts
- `LLMBasedAnswerCorrectness`: Overall correctness of the Generated Answer based on the Question and Ground Truth Answer(s)
- `LLMBasedAnswerRelevance`: Relevance of the Generated Answer w.r.t the Question
- `LLMBasedStyleConsistency`: Consistency of style bwtween the Generated Answer and the Ground Truth Answer(s)


## Resources

- **Docs:** [link](https://docs.relari.ai/)
- **Blog Post: Practical Guide to RAG Pipeline Evaluation:** [Part 1: Retrieval](https://medium.com/relari/a-practical-guide-to-rag-pipeline-evaluation-part-1-27a472b09893), [Part 2: Generation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d)
- **Discord:** Join our community of LLM developers [Discord](https://discord.gg/GJnM8SRsHr)
- **Reach out to founders:** [Email](mailto:founders@relari.ai) or [Schedule a chat](https://cal.com/yizhang/continuous-eval)

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## Open Analytics

We monitor basic anonymous usage statistics to understand our users' preferences, inform new features, and identify areas that might need improvement.
You can take a look at exactly what we track in the [telemetry code](continuous_eval/utils/telemetry.py)

To disable usage-tracking you set the `CONTINUOUS_EVAL_DO_NOT_TRACK` flag to `true`.
