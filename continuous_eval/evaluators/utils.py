def validate_dataset(dataset):
    """
    Validate the dataset format.
    """
    assert isinstance(dataset, list), "Dataset must be a list."
    for item in dataset:
        assert isinstance(item, dict), "Each item in the dataset must be a dictionary."
        assert "question" in item
        assert "retrieved_contexts" in item
        assert "ground_truth_contexts" in item
        assert "ground_truths" in item
        assert len(item["retrieved_contexts"]) > 0
        assert len(item["ground_truth_contexts"]) > 0
        assert len(item["ground_truths"]) > 0
        assert isinstance(item["retrieved_contexts"], list)
        assert isinstance(item["ground_truth_contexts"], list)
        assert isinstance(item["ground_truths"], list)
        assert isinstance(item["question"], str)
        assert isinstance(item["retrieved_contexts"][0], str)
        assert isinstance(item["ground_truth_contexts"][0], str)
        assert isinstance(item["ground_truths"][0], str)
