from eval_core.dataset import read_jsonl, write_jsonl


def test_round_trip_keeps_text_and_order(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "a", "text": "hello"}, {"id": "b", "text": "café नमस्ते"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    assert [r["id"] for r in read_jsonl(path)] == [1, 2]


def test_write_creates_missing_folder(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"id": 1}])
    assert path.exists()
