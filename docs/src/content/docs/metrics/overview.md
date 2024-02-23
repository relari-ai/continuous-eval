---
title: Overview of Metrics
description: Overview of different types of metrics
sidebar:
  badge:
    text: beta
    variant: tip
---

## Metric Categories

The `continuous-eval` package offers three categories of metrics based on how they are computed:

- **Deterministic metrics:** calculated based on statistical formulas
- **Semantic:** calculated using smaller models
- **LLM-based:** calculated by an Evaluation LLM with curated prompts

All the metrics comes with pros and cons and there's not a one-size-fits-all evaluation pipeline that's optimal for every use case. We aim to provide a wide range of metrics for you to choose from.

The package also offers a way to [**Ensemble Metrics**](/v0.3/metrics/ensembling_classifier/) of different metrics to improve performance on quality and effeciency.


<br>

### `Metric` Class 

Below is the list of metrics available:

<table>
    <tr>
        <th>Module</th>
        <th>Category</th>
        <th>Metrics</th>
    </tr>
    <tr>
        <td rowspan="2">Retrieval</td>
        <td>Deterministic</td>
        <td>PrecisionRecallF1, RankedRetrievalMetrics</td>
    </tr>
    <tr>
        <td>LLM-based</td>
        <td>LLMBasedContextPrecision, LLMBasedContextCoverage</td>
    </tr>
    <tr>
        <td rowspan="3">Text Generation</td>
        <td>Deterministic</td>
        <td>DeterministicAnswerCorrectness, DeterministicFaithfulness, FleschKincaidReadability</td>
    </tr>
    <tr>
        <td>Semantic</td>
        <td>DebertaAnswerScores, BertAnswerRelevance, BertAnswerSimilarity</td>
    </tr>
    <tr>
        <td>LLM-based</td>
        <td>LLMBasedFaithfulness, LLMBasedAnswerCorrectness, LLMBasedAnswerRelevance, LLMBasedStyleConsistency</td>
    </tr>
    <tr>
        <td rowspan="1">Classification</td>
        <td>Deterministic</td>
        <td>ClassificationAccuracy</td>
    </tr>
    <tr>
        <td rowspan="2">Code Generation</td>
        <td>Deterministic</td>
        <td>CodeStringMatch, PythonASTSimilarity</td>
    </tr>
    <tr>
        <td>LLM-based</td>
        <td>LLMBasedCodeGeneration</td>
    </tr>
    <tr>
        <td>Agent Tools</td>
        <td>Deterministic</td>
        <td>ToolSelectionAccuracy</td>
    </tr>
    <tr>
        <td>Custom</td>
        <td></td>
        <td>Define your own metrics</td>
    </tr>
</table>


#### Retrieval metrics

##### Deterministic

**`PrecisionRecallF1`**
- **Definition:** Rank-agnostic metrics including Precision, Recall, and F1 of Retrieved Contexts
- **Inputs:** `retrieved_contexts`, `ground_truth_contexts`

**`RankedRetrievalMetrics`**
- **Definition:** Rank-aware metrics including Mean Average Precision (MAP), Mean Reciprical Rank (MRR), NDCG (Normalized Discounted Cumulative Gain) of retrieved contexts
- **Inputs:** `retrieved_contexts`, `ground_truth_contexts`

##### LLM-based

**`LLMBasedContextPrecision`**
- **Definition:** Precision and Mean Average Precision (MAP) based on context relevancy classified by LLM
- **Inputs:** `question`, `retrieved_contexts`

**`LLMBasedContextCoverage`**
- **Definition:** Proportion of statements in ground truth answer that can be attributed to Retrieved Contexts calculated by LLM
- **Inputs:** `question`, `retrieved_contexts`, `ground_truth_answers`

#### Text Generation metrics

##### Deterministic

**`DeterministicAnswerRelevance`**
- **Definition:** Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- **Inputs:** `question`, `generated_answer`

**`DeterministicFaithfulness`**
- **Definition:** Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision, and BLEU score
- **Inputs:** `retrieved_contexts`, `generated_answer`

**`FleschKincaidReadability`**
- **Definition:** How easy or difficult it is to understand the LLM generated answer.
- **Inputs:** `generated_answer`

##### Semantic

**`DebertaAnswerScores`**
- **Definition:** Entailment and contradiction scores between the Generated Answer and Ground Truth Answer
- **Inputs:** `generated_answer`, `ground_truth_answers`

**`BertAnswerRelevance`**
- **Definition:** Similarity score based on the BERT model between the Generated Answer and Question
- **Inputs:** `question`, `generated_answer`

**`BertAnswerSimilarity`**
- **Definition:** Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer
- **Inputs:** `generated_answer`, `ground_truth_answers`

##### LLM-based

**`LLMBasedFaithfulness`**
- **Definition:** Binary classifications of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts by LLM
- **Inputs:** `question`, `retrieved_contexts`, `generated_answer`

**`LLMBasedAnswerCorrectness`**
- **Definition:** Overall correctness of the Generated Answer based on the Question and Ground Truth Answer calculated by LLM
- **Inputs:** `question`, `generated_answer`, `ground_truth_answers`

**`LLMBasedAnswerRelevance`**
- **Definition:** Relevance of the Generated Answer with respect to the Question
- **Inputs:** `question`, `generated_answer`

**`LLMBasedStyleConsistency`**
- **Definition:** Consistency of style between the Generated Answer and the Ground Truth Answer(s)
- **Inputs:** `generated_answer`, `ground_truth_answers`

#### Code Generation metrics

##### Deterministic

**`DeterministicAnswerRelevance`**
- **Definition:** Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- **Inputs:** `question`, `generated_answer`

**`DeterministicFaithfulness`**
- **Definition:** Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision, and BLEU score
- **Inputs:** `retrieved_contexts`, `generated_answer`

#### Classification metrics

##### Deterministic

**`ClassificationAccuracy`**
- **Definition:** Proportion of correctly identified items out of the total items
- **Inputs:** `predictions`, `ground_truth_labels`

#### Code Generation metrics

##### Deterministic

**`CodeStringMatch`**
- **Definition:** Exact and fuzzy match scores between generated code strings and the ground truth code strings
- **Inputs:** `answer`, `ground_truths`

**`PythonASTSimilarity`**
- **Definition:** Similarity of Abstract Syntax Trees (ASTs) for Python code, comparing the generated code to the ground truth code
- **Inputs:** `answer`, `ground_truths`

#### Agent Tools metrics

##### Deterministic

**`ToolSelectionAccuracy`**
- **Definition:** Accuracy of selecting the correct tool(s) for a given task by the agent
- **Inputs:** `tools`, `ground_truths`