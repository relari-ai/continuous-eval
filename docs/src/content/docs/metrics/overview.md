---
title: Overview of Metrics
description: Overview of different types of metrics
sidebar:
  badge:
    text: beta
    variant: tip
---

## What the Metrics Measure

:::note
In Retrieval-augmented Generation (RAG) pipelines, we seperate the evaluation process for Retrieval and Generation because they have **distinct functionality and requirements**.
:::

**For Retrieval, we care how well the system can fetch the relevent documents to answer the question.** Specifically, we try to measure:

- **Context Recall:** how completely has the system retrieved all the necessary documents
- **Context Precision:** how much signal (vs. noise) did the system retrieve?

**For Generation, we care about how well the LLM answers the question based on the retrieved contexts.** Specifically, we try to measure:

- **Overall Correctness:** how closely the answer match with an ideal reference answer?
- **Faithfulness:** how well is the answer grounded on context retrieved?
- **Relevance:** is the answer a direct response to the question?
- **Logic, Style, and many more aspects**

<br>

## Metric Categories

The `continuous-eval` package offers three categories of metrics based on how they are computed:

- **Deterministic metrics:** calculated based on statistical formulas
- **Semantic:** calculated using smaller models
- **LLM-based:** calculated by an Evaluation LLM with curated prompts

All the metrics comes with pros and cons and there's not a one-size-fits-all evaluation pipeline that's optimal for every use case. We aim to provide a wide range of metrics for you to choose from.

The package also offers a way to **Ensemble Metrics** of different metrics to improve performance on quality and effeciency.

:::tip
Check out our blog post that dives deeper into the pros and cons of different types of metrics:
**A Practical Guide to RAG Pipeline Evaluation:** [Part 1: Retrieval](https://medium.com/relari/a-practical-guide-to-rag-pipeline-evaluation-part-1-27a472b09893), [Part 2: Generation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d)
:::

<br>

### `Metric` Class 

Below is the list of metrics available:
#### Retrieval metrics

##### Deterministic

- `PrecisionRecallF1`: Rank-agnostic metrics including Precision, Recall, and F1 of Retrieved Contexts
- `RankedRetrievalMetrics`: Rank-aware metrics including Mean Average Precision (MAP), Mean Reciprical Rank (MRR), NDCG (Normalized Discounted Cumulative Gain) of retrieved contexts

##### LLM-based

- `LLMBasedContextPrecision`: Precision and Mean Average Precision (MAP) based on context relevancy classified by LLM
- `LLMBasedContextCoverage`: Proportion of statements in ground truth answer that can be attributed to Retrieved Contexts calcualted by LLM

#### Generation metrics

##### Deterministic

- `DeterministicAnswerRelevance`: Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- `DeterministicFaithfulness`: Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision and BLEU score

##### Semantic

- `DebertaAnswerScores`: Entailment and contradiction scores between the Generated Answer and Ground Truth Answer
- `BertAnswerRelevance`: Similarity score based on the BERT model between the Generated Answer and Question
- `BertAnswerSimilarity`: Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer

##### LLM-based

- `LLMBasedFaithfulness`: Binary classifications of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts by LLM
- `LLMBasedAnswerCorrectness`: Score (1-5) of the Generated Answer based on the Question and Ground Truth Answer calcualted by LLM


## Data Dependencies

<style>
  table {
    border-collapse: collapse;
    font-size: 14px;
    width: 100%; /* Optional: Adjust width as needed */
    table-layout: fixed; /* Optional: For equal column width */
  }

  th, td {
    padding: 10px;
    border: 1px solid #333; /* Darker border color */
  }

  th {
    background-color: #37474F; /* Soft dark blue-grey for headers */
    color: white; /* White text for contrast */
    font-weight: bold;
  }

  tr {
    background-color: transparent
  }

  .header-row {
    background-color: #007BFF; /* Deep blue for main headers */
    color: white;
    text-align: center;
  }

  .sub-header {
    background-color: #6c757d; /* Darker grey for sub-headers */
    color: white; /* White text for sub-headers */
    font-style: italic;
  }

  .check {
    text-align: center; /* Centering checkmark */
  }
  
  .check::before {
    content: '\2713'; /* Unicode for checkmark */
    color: green; /* Checkmark color */
  }

  .grey {
    text-align: center;
  }

  .grey::before {
    content: '-'; /* Unicode for checkmark */
    color: grey; /* Checkmark color */
  }

</style>

#### Input data for retrieval metrics

<table>

  <tr class="sub-header">
    <th colspan="2">Retrieval Metrics</th>
    <th>Question</th>
    <th>*Retrieved Contexts*</th>
    <th>Ground Truth Contexts</th>
    <th>Generated Answer</th>
    <th>Ground Truth Answers</th>
  </tr>
    <tr>
    <th rowspan="2">Deterministic</th>
    <td>Context Precision & Recall</td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <td>Rank-Aware Metrics</td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <th rowspan="2">LLM-Based</th>
    <td>Context Precision</td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <td>Context Coverage</td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
  </tr>
</table>

#### Input data for generation metrics


<table>
  <tr class="sub-header">
    <th colspan="2">Generation Metrics</th>
    <th>Question</th>
    <th>Retrieved Contexts</th>
    <th>Ground Truth Contexts</th>
    <th>*Generated Answer*</th>
    <th>Ground Truth Answers</th>
  </tr>
  <tr>
    <tr>
    <th rowspan="2">Deterministic</th>
    <td>Faithfulness</td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <td>Answer Relevance</td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
  <th rowspan="3">Semantic</th>
    <td>Bert Answer Similarity</td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
  </tr>
    <tr>
  <td>Bert Answer Relevance</td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <td>Deberta Answer Scores</td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
  </tr>
  <tr>
    <th rowspan="2">LLM-Based</th>
    <td>Correctness</td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
  </tr>
  <tr>
    <td>Faithfulness</td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
  </tr>


</table>

\*variable being evaluated\*