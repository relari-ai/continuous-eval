from pathlib import Path
from time import perf_counter

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import Dataset, EvaluationRunner, SingleModulePipeline
from continuous_eval.eval.tests import GreaterOrEqualThan
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

# Let's download the retrieval dataset example
dataset_jsonl = example_data_downloader("retrieval")
dataset = Dataset(dataset_jsonl).sample(10)

pipeline = SingleModulePipeline(
    dataset=dataset,
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
        GreaterOrEqualThan(test_name="Recall", metric_name="rouge_l_recall", min_value=0.8),
    ],
)

# We start the evaluation manager and run the metrics
tic = perf_counter()
runner = EvaluationRunner(pipeline)
eval_results = runner.evaluate()
toc = perf_counter()
print(eval_results.aggregate())
print(f"Elapsed time: {toc - tic:.2f} seconds")
