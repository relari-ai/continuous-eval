---
title: Overview
description: Overview of different types of metrics
---

## Data Dependencies

<!-- <style>
  table {
    border-collapse: collapse;
    font-size: 14px;
  }

  th, td {
    padding: 4px;
    border: 1px solid black;
    background-color: transparent;
  }

  .check {
    text-align: center; /* Centering checkmark */
  }
  
  .check::before {
    content: '\2713'; /* Unicode for checkmark */
  }

  .grey {
    background-color: lightgrey; /* Grey color for specific cells */
  }
</style> -->

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