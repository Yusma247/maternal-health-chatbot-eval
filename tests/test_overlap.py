from eval_core.overlap import best_match, max_overlap, overlap_score


def test_paraphrase_scores_at_or_above_threshold():
    a = "Can my pregnant patient take ibuprofen in the third trimester?"
    b = "Is ibuprofen safe for a patient in third trimester of pregnancy?"
    assert overlap_score(a, b) >= 0.5


def test_related_topic_but_different_question_scores_low():
    a = "What should I tell her about eating iron rich foods?"
    b = "My patient says iron tablets make the baby dark skinned. What should I tell her?"
    assert overlap_score(a, b) < 0.5


def test_very_short_text_scores_zero():
    assert overlap_score("Hello there", "Hello there friend") == 0.0


def test_best_match_returns_score_and_index():
    a = "Can my pregnant patient take ibuprofen in the third trimester?"
    others = ["Newborn feeding schedule at night time", a]
    assert best_match(a, others) == (1.0, 1)
    assert best_match(a, []) == (0.0, -1)


def test_max_overlap_takes_the_highest_and_handles_empty():
    a = "Can my pregnant patient take ibuprofen in the third trimester?"
    others = ["Newborn feeding schedule at night time", a]
    assert max_overlap(a, others) == 1.0
    assert max_overlap(a, []) == 0.0
