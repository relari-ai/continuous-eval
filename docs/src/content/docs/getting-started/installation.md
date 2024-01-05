---
title: Installation
description: How to install continuous-eval
---

`continuous-eval` is provided as an open-source Python package. 
To install it, run the following command:

```bash
python3 -m pip install continuous-eval
```

the package offers optional extras for additional functionality:

- `anthropic` to support Anthropic's Claude model
- `gemini` to support Google's Gemini model
- `generators` to support automatic dataset generation

with PIP you can install any combination of them with:

```bash
pip install continuous-eval[anthropic,gemini,generators]
```

Otherwise you can install continuous-eval from source

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
