---
title: Dataset Generator
sidebar:
  badge:
    text: beta
    variant: caution
---

**We recommend AI teams invest in manually curating a high-quality golden dataset** (created by users, or domain experts) to properly evaluate and improve the LLM pipeline.

Every (RAG-based) LLM application is different in functionalities and requirements, and the evaluation golden dataset should be diverse enough to capture different design requirements.

If you don't have a golden dataset, you can use `SimpleDatasetGenerator` to create a silver dataset as a starting point, upon which you can modify and improve.

### Simple Dataset Generator

The `SimpleDatasetGenerator` loads indicies from a vector database (using the Langchain interface) and samples select vectors to create questions.

The follow 4 types of questions can be created:

-   **Single-Hop Fact-Seeking Questions:** An information seeking question, answer pair created based on a single chunk
-   **Single-Hop Reasoning Questions:** A Why / How question, answer pair created based on a single chunk
-   **Multi-Hop Fact-Seeking Questions:** An information seeking question, answer pair created based on two chunks
-   **Multi-Hop Reasoning Questions:** A Why / How question, answer pair created based on a two chunks

There are multiple filtering processes. If the LLM fails to generate a question, extract the relevant sentences from the contexts, or generate a reasonable answer, the generation will fail. The pipeline will try up to `max_try_ratio` for Multi-Hop questions and fill the gap with Single-Hop questions.

### Example Usage

Below is an example of generating dataset

```python
import datetime
import logging
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.generators import SimpleDatasetGenerator
from continuous_eval.llm_factory import LLMFactory

load_dotenv()

def main():
    logging.basicConfig(level=logging.INFO)

    generator_llm = "gpt-4-1106-preview"
    num_questions = 10
    multi_hop_precentage = 0.25
    max_try_ratio = 3

    print(f"Generating a {num_questions}-questions dataset with {generator_llm}...")
    db = example_data_downloader("graham_essays/small/chromadb", Path("temp"), force_download=False)

    tic = perf_counter()
    dataset_generator = SimpleDatasetGenerator(
        vector_store_index=db,
        generator_llm=LLMFactory(generator_llm),
    )
    dataset = dataset_generator.generate(
        embedding_vector_size=1536,
        num_questions=num_questions,
        multi_hop_percentage=multi_hop_precentage,
        max_try_ratio=max_try_ratio,
    )
    toc = perf_counter()
    print(f"Finished generating dataset in {tic-toc:.2f}sec.")

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = Path("generated_dataset")
    output_directory.mkdir(parents=True, exist_ok=True)
    fname = (
        output_directory / f"G_{generator_llm}_Q_{num_questions}_MH%_{multi_hop_precentage}_{current_datetime}.jsonl"
    )
    print(f"Saving dataset to {fname}")
    dataset.to_jsonl(fname)
    print(f"Done.")


if __name__ == "__main__":
    main()
```
