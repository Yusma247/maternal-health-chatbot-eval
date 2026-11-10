from eval_core.genparse import parse_qa_json


def test_plain_json_array():
    text = '[{"question": "Q1", "answer": "A1"}]'
    assert parse_qa_json(text) == [{"question": "Q1", "answer": "A1"}]


def test_fenced_json_array():
    text = '```json\n[{"question": "Q1", "answer": "A1"}]\n```'
    assert len(parse_qa_json(text)) == 1


def test_garbage_gives_empty_list():
    assert parse_qa_json("Sorry, I cannot do that.") == []


def test_empty_array_gives_empty_list():
    assert parse_qa_json("[]") == []


def test_items_missing_keys_are_skipped():
    text = '[{"question": "Q1"}, {"question": "Q2", "answer": "A2"}]'
    assert parse_qa_json(text) == [{"question": "Q2", "answer": "A2"}]


def test_non_list_json_gives_empty_list():
    assert parse_qa_json('{"question": "Q", "answer": "A"}') == []
