from eval_core.passages import chunk_tokens


def test_chunks_overlap_and_cover_the_document():
    tokens = [(f"w{i}", 1) for i in range(1000)]
    chunks = chunk_tokens(tokens, size=400, overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 400
    assert chunks[1][0] == tokens[350]
    assert chunks[-1][-1] == tokens[-1]


def test_short_document_gives_no_chunks():
    assert chunk_tokens([("w", 1)] * 50) == []


def test_short_tail_is_dropped():
    tokens = [("w", 1)] * 420
    assert len(chunk_tokens(tokens, size=400, overlap=50)) == 1
