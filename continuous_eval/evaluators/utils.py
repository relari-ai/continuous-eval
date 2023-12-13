def validate_dataset(dataset):
    """
    Validate the dataset format.
    """
    assert isinstance(dataset, list), "Dataset must be a list."
    NO_GROUND_TRUTH_CONTEXTS = False
    NO_GROUND_TRUTH_ANSWERS = False
    for item in dataset:
        assert isinstance(item, dict), "Each item in the dataset must be a dictionary."
        assert "question" in item
        assert isinstance(item["question"], str)

        assert "retrieved_contexts" in item
        assert len(item["retrieved_contexts"]) > 0
        assert isinstance(item["retrieved_contexts"], list)
        assert isinstance(item["retrieved_contexts"][0], str)

        assert "answer" in item
        assert isinstance(item["answer"], str)

        if "ground_truth_contexts" in item:
            assert len(item["ground_truth_contexts"]) > 0
            assert isinstance(item["ground_truth_contexts"], list)
            assert isinstance(item["ground_truth_contexts"][0], str)
        else:
            NO_GROUND_TRUTH_CONTEXTS = True

        if "ground_truths" in item:
            assert len(item["ground_truths"]) > 0
            assert isinstance(item["ground_truths"], list)
            assert isinstance(item["ground_truths"][0], str)
        else:
            NO_GROUND_TRUTH_ANSWERS = True
            print(
                "Warning: ground_truths (answer) not found in the dataset. Cannot run LLM_based context coverage, answer correctness, and answer similarity metrics."
            )

    if NO_GROUND_TRUTH_CONTEXTS:
        print(
            "Warning: ground_truth_contexts not found for every item in the dataset. Cannot run deterministic retrieval metrics."
        )
    if NO_GROUND_TRUTH_ANSWERS:
        print(
            "Warning: ground_truths (answer) not found for every item in the dataset. Cannot run LLM_based context coverage, answer correctness, and answer similarity metrics."
        )
