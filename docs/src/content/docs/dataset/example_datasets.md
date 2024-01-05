---
title: Example Datasets
---

### Datasets Available


Below are the example datasets you can use to 

<table>
  <tr>
    <th>Dataset</th>
    <th>Description</th>
    <th>Data format</th>
  </tr>
  <tr>
    <td>correctness</td>
    <td>1,200 examples, created from <a href="https://github.com/McGill-NLP/instruct-qa">InstructQA</a></td>
    <td>`Dataset`</td>
  </tr>
  <tr>
    <td>retrieval</td>
    <td>300 examples, created from <a href="https://hotpotqa.github.io/">HotpotQA</a></td>
    <td>`Dataset`</td>
  </tr>
  <tr>
    <td>faithfulness</td>
    <td>544 examples, created from <a href="https://github.com/McGill-NLP/instruct-qa">InstructQA</a></td>
    <td>`Dataset`</td>
  </tr>
  <tr>
    <td>graham_essays/small/txt</td>
    <td>10 Paul Graham essays, created from <a href="https://github.com/ofou/graham-essays">graham-essays</a></td>
    <td>Zip of txt</td>
  </tr>
  <tr>
    <td>graham_essays/small/chromadb</td>
    <td>OpenAI Embeddings of 395 chunks from 10 Paul Graham essays</td>
    <td>Zip of embeddings (in ChromaDB format)</td>
  </tr>
</table>

### Download Datasets


The example datasets can be `example_data_downloader` helper function.

```python
from continuous_eval.data_downloader import example_data_downloader

# Download a dataset for evaluation
dataset = example_data_downloader("retrieval")

# Download embeddings for dataset generation
db = example_data_downloader("graham_essays/small/chromadb", Path("temp"), force_download=False)
```