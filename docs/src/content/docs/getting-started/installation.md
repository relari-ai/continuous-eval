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

- `semantic` to support semantic metrics that use small models such as BERT, DeBERTa.
- `anthropic` to support Anthropic's Claude model
- `google` to support Google's Gemini model
- `bedrock` to support AWS's Bedrock models
- `cohere` to support Cohere's models
- `azure` to support Microsoft's Azure models

with PIP you can install any combination of them. For example:

```bash
pip install continuous-eval[anthropic,gemini,generators]
```

Otherwise you can install continuous-eval from source

```bash
git clone https://github.com/relari-ai/continuous-eval.git && cd continuous-eval
poetry install --all-extras
```

continuous-eval is tested on Python 3.10, 3.11 and 3.12 on Linux and MacOS.
