import argparse
import csv
import time
from pathlib import Path

from eval_core.checks import check_forbidden, check_required
from eval_core.dataset import read_jsonl
from eval_core.judge import judge
from eval_core.models import ask_gemini, ask_ollama
from eval_core.prompts import DIDI_SYSTEM
from eval_core.summary import format_table, summarize

FIELDS = [
    "id", "prompt", "answer", "required_passed", "required_total",
    "violations", "violation_text", "judge_score", "seconds",
]


def load_answers(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["answer"] for row in csv.DictReader(f)}


def get_answer(args, case, answers):
    if args.backend == "file":
        return answers[case["id"]]
    if args.backend == "ollama":
        return ask_ollama(case["prompt"], DIDI_SYSTEM, args.model)
    return ask_gemini(case["prompt"], DIDI_SYSTEM, model=args.model or "gemini-2.5-flash")


def main():
    parser = argparse.ArgumentParser(description="Score a model on the test set")
    parser.add_argument("--backend", choices=["ollama", "gemini", "file"], required=True)
    parser.add_argument("--model", default="", help="Ollama tag or Gemini model name")
    parser.add_argument("--answers", default="", help="CSV with id,answer (file backend)")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    cases = read_jsonl(args.test)
    answers = load_answers(args.answers) if args.backend == "file" else {}
    rows = []
    for case in cases:
        start = time.time()
        answer = get_answer(args, case, answers)
        seconds = round(time.time() - start, 2)
        required = check_required(answer, case.get("must_include", []))
        forbidden = check_forbidden(answer, case.get("must_not", []))
        score = None if args.no_judge else judge(case["prompt"], answer)
        row = {
            "id": case["id"],
            "prompt": case["prompt"],
            "answer": answer,
            "required_passed": required["passed"],
            "required_total": required["total"],
            "violations": len(forbidden["violations"]),
            "violation_text": " | ".join(v["sentence"] for v in forbidden["violations"]),
            "judge_score": "" if score is None else round(score, 1),
            "seconds": seconds,
        }
        rows.append(row)
        print(
            f"{row['id']}: required {row['required_passed']}/{row['required_total']}, "
            f"forbidden {row['violations']}, judge {row['judge_score']}, {seconds}s"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(format_table({args.model or args.backend: summarize(rows)}))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
