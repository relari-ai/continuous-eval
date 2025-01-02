from time import perf_counter

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import EvaluationRunner, SingleModulePipeline
from continuous_eval.eval.tests import GreaterOrEqualThan
from continuous_eval.metrics.retrieval import (
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


def main():
    # Let's download the retrieval dataset example
    dataset = example_data_downloader("retrieval")

    pipeline = SingleModulePipeline(
        dataset=dataset,  # type: ignore
        eval=[
            PrecisionRecallF1().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
            RankedRetrievalMetrics().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
        ],
        tests=[
            GreaterOrEqualThan(
                test_name="Recall", metric_name="context_recall", min_value=0.8
            ),
        ],
    )

    # We start the evaluation manager and run the metrics
    tic = perf_counter()
    runner = EvaluationRunner(pipeline)
    eval_results = runner.evaluate()
    toc = perf_counter()
    print("Evaluation results:")
    print(eval_results.aggregate())
    print(f"Elapsed time: {toc - tic:.2f} seconds\n")

    print("Running tests...")
    test_results = runner.test(eval_results)
    print(test_results)


if __name__ == "__main__":
    # It is important to run this script in a new process to avoid
    # multiprocessing issues
    main()
