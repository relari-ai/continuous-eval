---
title: Getting started
description: How to install continuous-eval
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

**Optional:**
If you want to run LLM-based metrics, continuous-eval supports OpenAI, Anthropic and Google models, which require API keys:

- `OPENAI_API_KEY` 
- `ANTHROPIC_API_KEY` (optional)
- `GEMINI_API_KEY` (optional)

```bash
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
```