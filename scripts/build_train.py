import argparse
import csv

from eval_core.dataset import read_jsonl, write_jsonl
from eval_core.overlap import best_match
from eval_core.prompts import DIDI_SYSTEM

MIN_EXAMPLES = 50


def main():
    parser = argparse.ArgumentParser(description="Build train.jsonl from reviewed drafts")
    parser.add_argument("--draft", default="data/draft_qa.csv")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--drop-at", type=float, default=0.7,
                        help="Drop a question this similar to a test prompt")
    parser.add_argument("--review-at", type=float, default=0.5,
                        help="Print a warning from this similarity up")
    args = parser.parse_args()

    cases = read_jsonl(args.test)
    test_prompts = [case["prompt"] for case in cases]
    kept, dropped, to_review = [], 0, 0
    with open(args.draft, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("keep", "").strip() != "1":
                continue
            score, index = best_match(row["question"], test_prompts)
            if score >= args.drop_at:
                print(f"DROPPED (overlap {score:.2f} with {cases[index]['id']}): {row['question']}")
                dropped += 1
                continue
            if score >= args.review_at:
                print(f"REVIEW  (overlap {score:.2f} with {cases[index]['id']}): {row['question']}")
                to_review += 1
            kept.append({
                "messages": [
                    {"role": "system", "content": DIDI_SYSTEM},
                    {"role": "user", "content": row["question"].strip()},
                    {"role": "assistant", "content": row["answer"].strip()},
                ],
                "passage_id": row["passage_id"],
            })

    write_jsonl(args.out, kept)
    print(f"Kept {len(kept)}, dropped {dropped}, flagged {to_review} for your review -> {args.out}")
    if len(kept) < MIN_EXAMPLES:
        print(f"Warning: fewer than {MIN_EXAMPLES} examples. Review more drafts.")


if __name__ == "__main__":
    main()
