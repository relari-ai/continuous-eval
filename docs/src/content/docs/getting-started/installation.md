---
title: Getting started
description: A guide in my new Starlight docs site.
---

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

continuous-eval is tested on Python 3.9 and 3.11.

continuous-eval supports OpenAI, Anthropic Claude and Google Cloud AI Platform Prediction.
It requires the API keys:

- `OPENAI_API_KEY` 
- `ANTHROPIC_API_KEY` (optional)
- `GOOGLE_API_KEY` (optional)
