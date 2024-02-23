from pathlib import Path

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import Dataset, SingleModulePipeline
from continuous_eval.eval.manager import eval_manager
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

# Let's download the retrieval dataset example
dataset_jsonl = example_data_downloader("retrieval")
dataset = Dataset(dataset_jsonl)

pipeline = SingleModulePipeline(
    dataset=dataset,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=dataset.retrieved_context,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=dataset.retrieved_context,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
    ],
)

# We start the evaluation manager and run the metrics
eval_manager.set_pipeline(pipeline)
eval_manager.evaluation.results = dataset.data
eval_manager.run_metrics()
eval_manager.metrics.save(Path("metrics_results_retr.json"))

print(eval_manager.metrics.aggregate())
