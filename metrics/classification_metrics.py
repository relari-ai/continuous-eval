class ClassificationAccuracy:
    def __call__(self, predicted_class, ground_truth_class):
        if predicted_class == ground_truth_class:
            return {'classification_correctness': 1.0}
        else:
            return {'classification_correctness': 0.0}