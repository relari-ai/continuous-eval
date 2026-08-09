from continuous_eval.metrics.base.response_type import Integer


def test_integer_parses_negative_scores_on_signed_range():
    """A judge response on a signed Integer scale must keep the sign.

    Integer supports negative ranges (e.g. a -2..2 polarity scale), but the
    numeric matcher previously used ``\\d+`` and dropped a leading minus, so a
    judge output of "-2" was read as +2 - a silent sign flip on exactly the
    scores that carry the most meaning.
    """
    scale = Integer(-2, 2)
    assert scale.score("-2") == -2
    assert scale.score("The sentiment is -1") == -1
    assert scale.score("score: -2 out of 2") == -2
    # positives and in-range values are unchanged
    assert scale.score("2") == 2
    assert scale.score("1") == 1


def test_integer_clamps_out_of_range_negative_to_floor():
    """On a non-negative scale, an out-of-range negative clamps to the floor
    rather than being mis-read as a positive value."""
    scale = Integer(0, 10)
    assert scale.score("-3") == 0
    assert scale.score("7") == 7
