from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import MatchingStrategy, PrecisionRecallF1, RankedRetrievalMetrics

retrieval = example_data_downloader("retrieval")

evaluator = RetrievalEvaluator(
    dataset=retrieval,
    metrics=[
        PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
        RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
    ],
)
evaluator.run(k=2)
print(evaluator.aggregated_results)
