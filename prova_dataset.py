from continuous_eval.dataset import Dataset
from continuous_eval.evaluators import RetrievalEvaluator, GenerationEvaluator
from continuous_eval.metrics import (
    MatchingStrategy,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
    DeterministicAnswerRelevance,
    DeterministicFaithfulness,
)

# correctness = Dataset.from_jsonl('data/correctness.jsonl')
# print(correctness.info())
# retrieval = Dataset.from_jsonl("data/retrieval.jsonl")
# print(retrieval.info())
faithfulness = Dataset.from_jsonl('data/faithfulness.jsonl')
print(faithfulness.info())

# invalid = Dataset.from_jsonl('data/invalid.jsonl')
# print(invalid.info())

# evaluator = RetrievalEvaluator(
#     dataset=retrieval,
#     metrics=[
#         PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
#         RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
#     ],
# )
# evaluator.run(k=2)
# print(evaluator.aggregated_results)

evaluator = GenerationEvaluator(
    dataset=faithfulness,
    metrics=[
        DeterministicFaithfulness(),
    ]
)
evaluator.run()
print(evaluator.aggregated_results)
evaluator.save('faithfulness_results.jsonl')