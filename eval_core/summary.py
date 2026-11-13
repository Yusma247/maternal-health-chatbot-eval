def summarize(rows):
    required_passed = sum(int(r["required_passed"]) for r in rows)
    required_total = sum(int(r["required_total"]) for r in rows)
    scores = [float(r["judge_score"]) for r in rows if str(r.get("judge_score", "")).strip() != ""]
    seconds = [float(r["seconds"]) for r in rows if str(r.get("seconds", "")).strip() != ""]
    return {
        "cases": len(rows),
        "required_rate": required_passed / required_total if required_total else 0.0,
        "cases_with_violations": sum(1 for r in rows if int(r["violations"]) > 0),
        "mean_judge": sum(scores) / len(scores) if scores else None,
        "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
    }


def format_table(named_summaries):
    lines = [
        "| Model | Cases | Required items passed | Cases with forbidden content | Mean judge score | Mean seconds |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in named_summaries.items():
        judge = "n/a" if s["mean_judge"] is None else f"{s['mean_judge']:.1f}"
        secs = "n/a" if s["mean_seconds"] is None else f"{s['mean_seconds']:.1f}"
        lines.append(
            f"| {name} | {s['cases']} | {s['required_rate'] * 100:.0f}% | "
            f"{s['cases_with_violations']} | {judge} | {secs} |"
        )
    return "\n".join(lines)


def worst_cases(rows, n=3):
    def badness(row):
        missing = int(row["required_total"]) - int(row["required_passed"])
        return (int(row["violations"]), missing)

    return sorted(rows, key=badness, reverse=True)[:n]
