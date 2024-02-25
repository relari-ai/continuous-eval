def test_classification_accuracy():
    metric = ClassificationAccuracy()
    assert metric('class1', 'class1')['classification_correctness'] == 1.0
    assert metric('class1', 'class2')['classification_correctness'] == 0.0
    assert metric(None, 'class1')['classification_correctness'] == 0.0
    assert metric('', 'class1')['classification_correctness'] == 0.0
    assert metric('class1', None)['classification_correctness'] == 0.0
    assert metric('class1', '')['classification_correctness'] == 0.0