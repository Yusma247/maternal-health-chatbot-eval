from eval_core.dataset import read_jsonl


def test_every_case_has_the_required_shape():
    cases = read_jsonl("data/test.jsonl")
    assert 5 <= len(cases) <= 15
    for case in cases:
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["prompt"], str) and len(case["prompt"].split()) >= 5
        assert isinstance(case["must_include"], list) and case["must_include"]
        for group in case["must_include"]:
            assert isinstance(group, list) and group
            assert all(isinstance(p, str) and p for p in group)
        assert isinstance(case["must_not"], list)
        assert all(isinstance(p, str) and p for p in case["must_not"])


def test_ids_are_unique():
    ids = [c["id"] for c in read_jsonl("data/test.jsonl")]
    assert len(ids) == len(set(ids))
