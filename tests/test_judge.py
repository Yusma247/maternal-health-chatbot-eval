from eval_core.judge import extract_rating


def test_plain_rating():
    assert extract_rating("Rating: 7/10\nReason: fine") == 70.0


def test_bold_rating():
    assert extract_rating("**Rating:** 8/10") == 80.0


def test_decimal_rating():
    assert extract_rating("Rating: 7.5/10") == 75.0


def test_missing_rating_returns_none():
    assert extract_rating("The answer looks fine.") is None
