from eval_core.summary import format_table, summarize, worst_cases


def make_row(case_id, passed, total, violations, judge, seconds):
    return {
        "id": case_id,
        "required_passed": str(passed),
        "required_total": str(total),
        "violations": str(violations),
        "judge_score": str(judge),
        "seconds": str(seconds),
    }


ROWS = [
    make_row("t01", 2, 2, 0, 80, 4.0),
    make_row("t02", 1, 2, 1, 40, 6.0),
]


def test_summarize_reads_string_values():
    result = summarize(ROWS)
    assert result["cases"] == 2
    assert result["required_rate"] == 0.75
    assert result["cases_with_violations"] == 1
    assert result["mean_judge"] == 60.0
    assert result["mean_seconds"] == 5.0


def test_summarize_handles_missing_judge_scores():
    rows = [make_row("t01", 1, 1, 0, "", 2.0)]
    assert summarize(rows)["mean_judge"] is None


def test_format_table_has_header_and_one_row_per_model():
    table = format_table({"base": summarize(ROWS), "tuned": summarize(ROWS)})
    lines = table.splitlines()
    assert lines[0].startswith("| Model")
    assert len(lines) == 4
    assert "| base | 2 | 75% | 1 | 60.0 | 5.0 |" in lines[2]


def test_worst_cases_puts_violations_first():
    assert worst_cases(ROWS, n=1)[0]["id"] == "t02"
