def chunk_tokens(tokens, size=400, overlap=50, min_size=80):
    """Split a list of tokens into overlapping chunks. Drops a short tail."""
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + size]
        if len(piece) < min_size:
            break
        chunks.append(piece)
    return chunks
