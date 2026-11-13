import argparse
from pathlib import Path

from pypdf import PdfReader

from eval_core.dataset import write_jsonl
from eval_core.passages import chunk_tokens


def read_tokens(pdf_path, first_page, last_page):
    reader = PdfReader(str(pdf_path))
    last = last_page or len(reader.pages)
    tokens = []
    for page_no in range(first_page, last + 1):
        text = reader.pages[page_no - 1].extract_text() or ""
        tokens.extend((word, page_no) for word in text.split())
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Split a PDF into passages")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=0, help="0 means to the end")
    parser.add_argument("--out-dir", default="data/passages")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    tokens = read_tokens(pdf, args.first_page, args.last_page)
    rows = []
    for index, piece in enumerate(chunk_tokens(tokens), start=1):
        rows.append({
            "id": f"{pdf.stem}-{index:03d}",
            "source": pdf.name,
            "first_page": piece[0][1],
            "last_page": piece[-1][1],
            "text": " ".join(word for word, _ in piece),
        })
    out = Path(args.out_dir) / f"{pdf.stem}.jsonl"
    write_jsonl(out, rows)
    print(f"{len(rows)} passages from {len(tokens)} words -> {out}")


if __name__ == "__main__":
    main()
