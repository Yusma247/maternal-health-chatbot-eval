import argparse
import csv
import glob
import time
from pathlib import Path

from eval_core.dataset import read_jsonl
from eval_core.genparse import parse_qa_json
from eval_core.models import ask_gemini, load_gemini_keys

GEN_SYSTEM = (
    "You write training data for a maternal and newborn health assistant used by "
    "ASHA workers in rural India. You only use facts from the passage you are given."
)

GEN_PROMPT = """Using ONLY the passage below, write {n} questions an ASHA worker might ask, with answers.

Rules for every answer:
- Start with "Action:" and one line saying what to do first.
- Then "Steps:" with 2 to 4 short bullets.
- Then "Danger signs:" with bullets, only if the passage lists them. Otherwise write "Danger signs: none listed".
- Then "Refer:" saying when to send to a doctor or hospital, only if the passage says so. Otherwise write "Refer: not stated".
- Use simple English. Do not add facts, numbers or doses that are not in the passage.
- If the passage cannot support a question, skip it.

Return ONLY a JSON array like [{"question": "...", "answer": "..."}]. Return [] if nothing fits.

Passage:
{passage}
"""

FIELDS = ["id", "passage_id", "question", "answer", "keep", "passage"]


def main():
    parser = argparse.ArgumentParser(description="Draft Q&A pairs from passages")
    parser.add_argument("--passages-glob", default="data/passages/*.jsonl")
    parser.add_argument("--per-passage", type=int, default=3)
    parser.add_argument("--max-passages", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", default="data/draft_qa.csv")
    parser.add_argument(
        "--model", default="",
        help="Gemini model override. Also pins calls to the last key in "
             "API_keys.json instead of cycling, since a model override "
             "usually means a different key's project is being targeted.",
    )
    args = parser.parse_args()

    ask_kwargs = {}
    if args.model:
        ask_kwargs["model"] = args.model
        ask_kwargs["api_key"] = load_gemini_keys()[-1]

    existing_rows = []
    if Path(args.out).exists():
        with open(args.out, newline="", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
    covered_passage_ids = {r["passage_id"] for r in existing_rows}
    next_id = len(existing_rows) + 1

    passages = []
    for path in sorted(glob.glob(args.passages_glob)):
        passages.extend(read_jsonl(path))
    step = max(1, len(passages) // args.max_passages)
    passages = passages[::step][: args.max_passages]
    passages = [p for p in passages if p["id"] not in covered_passage_ids]
    print(f"Using {len(passages)} passages "
          f"({len(covered_passage_ids)} already covered in {args.out}, skipped)")

    new_rows = []
    for passage in passages:
        prompt = GEN_PROMPT.replace("{n}", str(args.per_passage)).replace("{passage}", passage["text"])
        try:
            reply = ask_gemini(prompt, GEN_SYSTEM, temperature=0.3, **ask_kwargs)
        except Exception as error:
            print(f"{passage['id']}: failed ({error})")
            continue
        pairs = parse_qa_json(reply)
        print(f"{passage['id']}: {len(pairs)} pairs")
        for pair in pairs:
            new_rows.append({
                "id": f"q{next_id + len(new_rows):04d}",
                "passage_id": passage["id"],
                "question": pair["question"],
                "answer": pair["answer"],
                "keep": "",
                "passage": passage["text"],
            })
        time.sleep(args.sleep)

    all_rows = existing_rows + new_rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(new_rows)} new drafts ({len(all_rows)} total) -> {args.out}")


if __name__ == "__main__":
    main()
