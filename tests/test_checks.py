from eval_core.checks import check_forbidden, check_required


def test_required_all_groups_present():
    groups = [["hospital", "doctor"], ["pressure"]]
    result = check_required("Apply firm pressure and go to the hospital now.", groups)
    assert result == {"passed": 2, "total": 2, "missing": []}


def test_required_reports_missing_group():
    groups = [["hospital", "doctor"], ["pressure"]]
    result = check_required("Go to the hospital now.", groups)
    assert result["passed"] == 1
    assert result["missing"] == [["pressure"]]


def test_required_is_case_insensitive():
    assert check_required("HOSPITAL", [["hospital"]])["passed"] == 1


def test_forbidden_flags_plain_claim():
    result = check_forbidden("Ibuprofen is safe in pregnancy.", ["ibuprofen is safe"])
    assert len(result["violations"]) == 1


def test_forbidden_ignores_negated_sentence():
    text = "Do not tell her that ibuprofen is safe in pregnancy."
    assert check_forbidden(text, ["ibuprofen is safe"])["violations"] == []


def test_negation_in_other_sentence_does_not_excuse_claim():
    text = "Avoid alcohol. Ibuprofen is safe."
    assert len(check_forbidden(text, ["ibuprofen is safe"])["violations"]) == 1


def test_forbidden_handles_newlines_as_sentence_breaks():
    text = "Never give it.\nIbuprofen is safe"
    assert len(check_forbidden(text, ["ibuprofen is safe"])["violations"]) == 1


def test_forbidden_ignores_not_to_phrasing():
    text = "Advise the mother not to give paracetamol syrup to the baby."
    assert check_forbidden(text, ["give paracetamol"])["violations"] == []
