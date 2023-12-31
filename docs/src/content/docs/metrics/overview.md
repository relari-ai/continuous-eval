---
title: Overview
description: Overview of different types of metrics
sidebar:
  order: -1
---

## What are we measuring

Retrieval-augmented Generation
We break down the pipeline into retrieval and generation because the two components have distinct functionality and requirements.
In retrieval, we care how well the system can fetch the relevent documents to answer the question. Specifically, we care about:

- Recall: how completely has the system retrieved all the necessary documents (More important metric!)
- Precision: how much signal (vs. noise) did the system retrieve?

In generation, we care about how well the LLM answers the question.
To break down:

- Overall Correctness: how closely the answer match with an ideal reference answer?
- Faithfulness: how well is the answer grounded on context retrieved?
- Relevance: is the answer a direct response to the question?
- Logic / Style / and many more aspects

Check out our blog post below that dives deeper:
**A Practical Guide to RAG Pipeline Evaluation:** [Part 1: Retrieval](https://medium.com/relari/a-practical-guide-to-rag-pipeline-evaluation-part-1-27a472b09893), [Part 2: Generation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d)

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


<table>
  <tr class="header-row">
    <th rowspan="2" colspan="2"></th>
    <th colspan="5">Input Data</th>
  </tr>
  <tr class="sub-header">
    <th>Question</th>
    <th>Retrieved Contexts</th>
    <th>Ground Truth Contexts</th>
    <th>Generated Answer</th>
    <th>Ground Truth Answers</th>
  </tr>
  <tr class="sub-header">
    <th colspan="7" style="text-align: left; font-style: italic;">Retrieval Metrics</th>
  <tr>
    <tr>
    <th rowspan="2">Deterministic</th>
    <td>Precision, Recall, F1</td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
  </tr>
  <tr>
    <td>Rank-aware Metrics</td>
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
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
  </tr>
<tr class="sub-header">
    <th colspan="7" style="text-align: left; font-style: italic;">Generation Metrics</th>
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
    <th rowspan="2">Semantic</th>
    <td>Bert Answer Similarity</td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
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
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "check"></td>
  </tr>
  <tr>
    <td>Faithfulness</td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
    <td class = "check"></td>
    <td class = "grey"></td>
  </tr>


</table>