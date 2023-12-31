---
title: Generation metrics
description: Overview of different types of metrics
---

### Deterministic

- `DeterministicAnswerRelevance`: Includes Token Overlap (Precision, Recall, F1), ROUGE-L (Precision, Recall, F1), and BLEU score of Generated Answer vs. Ground Truth Answer
- `DeterministicFaithfulness`: Proportion of sentences in Answer that can be matched to Retrieved Contexts using ROUGE-L precision, Token Overlap precision and BLEU score

### Semantic

- `DebertaAnswerScores`: Entailment and contradiction scores between the Generated Answer and Ground Truth Answer
- `BertAnswerRelevance`: Similarity score based on the BERT model between the Generated Answer and Question
- `BertAnswerSimilarity`: Similarity score based on the BERT model between the Generated Answer and Ground Truth Answer

### LLM-based

- `LLMBasedFaithfulness`: Binary classifications of whether the statements in the Generated Answer can be attributed to the Retrieved Contexts by LLM
- `LLMBasedAnswerCorrectness`: Score (1-5) of the Generated Answer based on the Question and Ground Truth Answer calcualted by LLM
