from typing import Any, Dict, List

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import (
    Dataset,
    EvaluationRunner,
    Module,
    ModuleOutput,
    Pipeline,
)
from continuous_eval.eval.result_types import PipelineResults
from continuous_eval.metrics.generation.text import (
    AnswerCorrectness,
)
from continuous_eval.metrics.retrieval import (
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


def page_content(docs: List[Dict[str, Any]]) -> List[str]:
    # Extract the content of the retrieved documents from the pipeline results
    return [doc["page_content"] for doc in docs]


def main():
    dataset: Dataset = example_data_downloader("graham_essays/small/dataset")  # type: ignore
    results: Dict = example_data_downloader("graham_essays/small/results")  # type: ignore

    # Simple 3-step RAG pipeline with Retriever->Reranker->Generation
    retriever = Module(
        name="retriever",
        input=dataset.question,  # type: ignore
        output=List[str],
        eval=[
            PrecisionRecallF1().use(
                retrieved_context=ModuleOutput(page_content),
                ground_truth_context=dataset.ground_truth_context,  # type: ignore
            ),
        ],
    )

    reranker = Module(
        name="reranker",
        input=retriever,
        output=List[Dict[str, str]],
        eval=[
            RankedRetrievalMetrics().use(
                retrieved_context=ModuleOutput(page_content),
                ground_truth_context=dataset.ground_truth_context,  # type: ignore
            ),
        ],
    )

    llm = Module(
        name="llm",
        input=reranker,
        output=str,
        eval=[
            AnswerCorrectness().use(
                question=dataset.question,  # type: ignore
                answer=ModuleOutput(),
                ground_truth_answers=dataset.ground_truth_answers,  # type: ignore
            ),
        ],
    )

    pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)
    print(pipeline.graph_repr())  # visualize the pipeline in marmaid format

    runner = EvaluationRunner(pipeline)
    eval_results = runner.evaluate(PipelineResults.from_dict(results))
    print(eval_results.aggregate())


if __name__ == "__main__":
    main()
