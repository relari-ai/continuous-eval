---
title: Introduction
description: Overview
sidebar:
  badge:
    text: new
    variant: tip
---

## What is continuous-eval?

`continuous-eval` is an open-source package created for the scientific and practical evaluation of LLM application pipelines. Currently, it focuses on retrieval-augmented generation (RAG) pipelines.

## Why another eval package?

Good LLM evaluation should help you reliably identify weaknesses in the pipeline, inform what actions to take, and accelerate your development from prototype to production. However, LLM evalulation today remains challenging because:

**Human evaluation is trustworthy but not scalable**
- Eyeballing can only be done on a small dataset, and it has to be repeated for any pipeline update  
- User feedback is spotty and lacks granularity

**Using LLMs to evaluate LLMs is expensive, slow and difficult to trust**
- Can be very costly and slow to run at scale
- Can be biased towards certain answers and often doesn’t align well with human evaluation

**Metrics don’t produce actions**
- End-to-end eval doesn’t reveal component-level performance
- It is difficult to turn metrics into “what’s next to try”

## How is continuous-eval different?

- **The Most Complete RAG Metric Library**: mix and match Deterministic, Semantic and LLM-based metrics.

- **Trustworthy Ensemble Metrics**: easily build a close-to-human ensemble evaluation pipeline with mathematical guarantees.

- **Cheaper and Faster Evaluation**: our hybrid pipeline slashes cost by up to 15x compared to pure LLM-based metrics, and reduces eval time on large datasets from hours to minutes.

- **Tailored to Your Data**: our evaluation pipeline can be customized to your use case and leverages the data you trust. We can help you curate a golden dataset if you don’t have one.

## Resources

- **Blog Post: Practical Guide to RAG Pipeline Evaluation:** [Part 1: Retrieval](https://medium.com/relari/a-practical-guide-to-rag-pipeline-evaluation-part-1-27a472b09893), [Part 2: Generation](https://medium.com/relari/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d)
- **Discord:** Join our community of LLM developers [Discord](https://discord.gg/GJnM8SRsHr)
- **Reach out to founders:** [Email](mailto:founders@relari.ai) or [Schedule a chat](https://cal.com/yizhang/continuous-eval)
