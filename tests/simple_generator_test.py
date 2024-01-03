import pytest

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.generators import SimpleDatasetGenerator
from continuous_eval.llm_factory import LLMFactory


def test_simple_generator():
    db = example_data_downloader("graham_essays/small/chromadb", force_download=False)
    dataset_generator = SimpleDatasetGenerator(
        vector_store_index=db,
        generator_llm=LLMFactory("gpt-3.5-turbo-1106"),
    )
    dataset = dataset_generator.generate(
        embedding_vector_size=1536,
        num_questions=3,
        multi_hop_percentage=0.5,
        max_try_ratio=5,
    )
    assert len(dataset) == 3
