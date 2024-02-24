from pathlib import Path

import numpy as np

from continuous_eval.classifiers import EnsembleMetric
from continuous_eval.classifiers.utils import eval_prediction
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.datatypes import DataSplit, SplitRatios
from continuous_eval.eval import Dataset, SingleModulePipeline
from continuous_eval.eval.manager import eval_manager
from continuous_eval.llm_factory import LLMFactory
from continuous_eval.metrics.generation.text import (
    DebertaAnswerScores,
    DeterministicAnswerCorrectness,
    FleschKincaidReadability,
    LLMBasedAnswerCorrectness,
)

# Download the correctness dataset and remove examples where the LLL refused to answer (i.e., said "I don't know")
dataset_jsonl = example_data_downloader("correctness")
dataset = Dataset(dataset_jsonl)
dataset.filter(lambda x: x["annotation"] != "refuse-to-answer")

# Let's define the system under evaluation
# We use a single module pipeline to evaluate the correctness of the answers
# using the DeterministicAnswerCorrectness metric, the DebertaAnswerScores metric, and the FleschKincaidReadability metric
# Attention: the DebertaAnswerScores metric requires the DeBERTa model (slow on CPU)
pipeline = SingleModulePipeline(
    dataset=dataset,
    eval=[
        DeterministicAnswerCorrectness().use(
            answer=dataset.answer, ground_truth_answers=dataset.ground_truths  # type: ignore
        ),
        DebertaAnswerScores().use(answer=dataset.answer, ground_truth_answers=dataset.ground_truths),  # type: ignore
        FleschKincaidReadability().use(answer=dataset.answer),  # type: ignore
    ],
)

# We start the evaluation manager and run the metrics
eval_manager.set_pipeline(pipeline)
eval_manager.evaluation.results = dataset.data
eval_manager.run_metrics()
eval_manager.metrics.save(Path("metrics_results.json"))

# Now we building the data for the ensemble classifier
# X is the input the classifier can use
# y is the target the classifier should predict (1 for correct, 0 for incorrect)
X = eval_manager.metrics.to_pandas()
y = map(lambda x: 1 if x == "correct" else 0, dataset["annotation"])

# We split the data into train, test, and calibration sets
# We also specify the features we want to use for the classifier
# We also oversample the train set to balance the classes
datasplit = DataSplit(
    X=X,
    y=y,
    split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
    features=[
        "token_overlap_recall",
        "deberta_answer_entailment",
        "deberta_answer_contradiction",
        "flesch_reading_ease",
    ],
    oversample=True,
)

# We use the train and calibration sets to train the classifier
predictor = EnsembleMetric(training=datasplit.train, calibration=datasplit.calibration)

# We then use the test set to evaluate the classifier
print("Running predictor (without judicator)")
y_hat, y_set = predictor.predict(datasplit.test.X)
num_undecided = np.sum(np.all(y_set, axis=1))
print(eval_prediction(datasplit.test.y, y_hat))
print(f"Undecided: {num_undecided} ({num_undecided/len(y_set):.2%})")

#######################################################################################
# Optional, use a judicator to resolve undecided examples
# Attention: the LLM model can be slow
#######################################################################################

print("\nRunning predictor (with judicator)")
llm_metric = LLMBasedAnswerCorrectness(LLMFactory("gpt-4-1106-preview"))


def judicator(idx):
    # The judicator receives the index of the example in the test set where the classifier is undecided
    # and in this case, since we are computing the correctness of the sample,
    # it returns True if the example is correct and False otherwise
    datum = dataset.data[idx]
    metric_result = llm_metric(
        question=datum["question"],
        answer=datum["answer"],
        ground_truth_answers=datum["ground_truths"],
    )
    return metric_result["LLM_based_answer_correctness"] >= 0.5


y_hat, _ = predictor.predict(datasplit.test.X, judicator=judicator)
print(eval_prediction(datasplit.test.y, y_hat))
