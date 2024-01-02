from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics, RougeChunkMatch, RougeSentenceMatch

retrieval = example_data_downloader("retrieval")

evaluator = RetrievalEvaluator(
    dataset=retrieval,
    metrics=[
        PrecisionRecallF1(RougeSentenceMatch()),
        RankedRetrievalMetrics(RougeChunkMatch()),
    ],
)
evaluator.run(k=2)
print(evaluator.aggregated_results)
