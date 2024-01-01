from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.metrics import PrecisionRecallF1, MatchingStrategy

retrieval = example_data_downloader("retrieval")
metric = PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH)
print(metric.calculate(**retrieval.datum(0)))