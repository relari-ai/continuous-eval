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

- `generators` to support automatic dataset generation
- `semantic` to support semantic metrics that use small models such as BERT, DeBERTa.
- `anthropic` to support Anthropic's Claude model
- `gemini` to support Google's Gemini model
- `bedrock` to support AWS's Bedrock models
- `cohere` to support Cohere's models
- `langchain` to enable some examples run with langchain_community

with PIP you can install any combination of them. For example:

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
If you want to run LLM-based metrics, continuous-eval supports a variety of models, which require API keys:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY` (optional)
- `GEMINI_API_KEY` (optional)
- Azure OpenAI API key (optional: `AZURE_ENDPOINT`, `AZURE_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_KEY`)
- `COHERE_API_KEY` (optional)

To bring your custom LLM endpoints using **vLLM** or **AWS Bedrock**, check out the guidance in [LLM Factory](https://github.com/relari-ai/continuous-eval/blob/main/continuous_eval/llm_factory.py) implementation.
