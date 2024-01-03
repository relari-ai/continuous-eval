---
title: Metric Ensembling
---

## Metric Ensembling

The aim of ensembling different metrics to predict the human label is to combine the strengths and balance out the weaknesses of individual metrics, ultimately leading to more accurate, robust, and reliable predictions. 

Each metric might capture different aspects of the data or be sensitive to different patterns, so when we combine them, we often get a more comprehensive view.

## What is Conformal Prediction?

Conformal Prediction is a statistical technique that quantifies the confidence level of a prediction. In this case, we are trying to predict whether the answer is correct (or faithful). With conformal prediction, instead of just saying “yes” (or “no”), the model tells us “the answer is correct with probability at least 90%”. In essence, conformal prediction doesn’t just give you an answer; it tells you how confident you can be in that answer. If the model is uncertain, conformal prediction will tell you it’s “undecided”. For the undecided datapoints, we ask the more powerful GPT-4 to judge its correctness.

## Paste in pictures from blog