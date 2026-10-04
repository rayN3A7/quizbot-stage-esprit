"""Analyse des questions : taux de réussite, discrimination, répartition des
choix, fidélité du quiz — et aucun signalement statistique sur trop peu de copies."""
from backend import analytics
from backend.models import Question, QuestionType, Quiz


def _mcq(key):
    return Question(type=QuestionType.MCQ, question="QCM ?", choices=["A", "B", "C", "D"],
                    correct_choice_index=key, reference_answer="")


def _open(reference="Réponse attendue"):
    return Question(type=QuestionType.OPEN, question="Ouverte ?", reference_answer=reference)


def _copy(*rows):
    """rows : (question, réponse, juste, points)"""
    return {"graded_answers": [{"question_id": q.id, "student_answer": a, "correct": ok, "score": s}
                               for q, a, ok, s in rows]}


def _codes(item):
    return [f["code"] for f in item["flags"]]


def test_a_miskeyed_question_stands_out_from_good_ones():
    good, miskeyed, open_a, open_b, easy = _mcq(1), _mcq(0), _open(), _open(), _open()
    quiz = Quiz(title="T", document_name="d.pdf", questions=[good, miskeyed, open_a, open_b, easy])
    strong = _copy((good, "1", True, 1.0), (miskeyed, "2", False, 0.0), (open_a, "juste", True, 1.0),
                   (open_b, "juste", True, 1.0), (easy, "juste", True, 1.0))

    def weak(good_answer, miskeyed_answer, miskeyed_ok):
        return _copy((good, good_answer, False, 0.0), (miskeyed, miskeyed_answer, miskeyed_ok, float(miskeyed_ok)),
                     (open_a, "faux", False, 0.0), (open_b, "faux", False, 0.0), (easy, "juste", True, 1.0))

    copies = [strong] * 3 + [weak("0", "0", True), weak("2", "0", True), weak("0", "3", False)]
    analysis = analytics.item_analysis(quiz, copies)
    items = analysis["items"]

    assert [i["success_rate"] for i in items] == [0.5, 0.333, 0.5, 0.5, 1.0]
    assert items[0]["discrimination"] > 0.5
    # Les trois meilleurs élèves choisissent C ; la réponse indiquée (A) l'est par les plus faibles.
    assert items[1]["discrimination"] < 0
    assert _codes(items[0]) == ["unused_distractor"]                       # D jamais choisi
    assert _codes(items[1]) == ["negative_discrimination", "unused_distractor", "distractor_beats_key"]
    assert [o["count"] for o in items[1]["choices"]["options"]] == [2, 0, 3, 1]
    assert _codes(items[2]) == _codes(items[3]) == []
    assert _codes(items[4]) == ["too_easy"]
    assert items[4]["discrimination"] is None                              # tout le monde réussit : indéfini
    assert analysis["students"] == 6 and analysis["reliability"] is not None


def test_hard_often_skipped_question_and_old_answers_stored_as_text():
    question = _mcq(0)
    quiz = Quiz(title="T", document_name="d.pdf", questions=[question])
    answers = [("", False), ("", False), ("1", False), ("B", False), ("0", True)]  # « B » : ancien format
    items = analytics.item_analysis(quiz, [_copy((question, a, ok, float(ok))) for a, ok in answers])["items"]

    assert items[0]["choices"]["options"][1]["count"] == 2 and items[0]["blank"] == 2
    assert _codes(items[0]) == ["too_hard", "distractor_beats_key", "unused_distractor", "unused_distractor",
                                "often_blank"]


def test_too_few_copies_give_numbers_but_no_statistical_flag():
    no_reference, mcq = _open(reference=""), _mcq(1)
    quiz = Quiz(title="T", document_name="d.pdf", questions=[no_reference, mcq])
    copies = [_copy((no_reference, "oui", False, 0.45), (mcq, "0", False, 0.0))] * 2
    analysis = analytics.item_analysis(quiz, copies)

    # Une question ouverte sans réponse attendue est un défaut de la question : signalée d'emblée.
    assert _codes(analysis["items"][0]) == ["no_reference_answer"]
    assert analysis["items"][1]["success_rate"] == 0.0 and _codes(analysis["items"][1]) == []
    assert analysis["items"][1]["discrimination"] is None and analysis["reliability"] is None


def test_reliability_is_one_for_identical_questions_and_undefined_for_constant_totals():
    a, b = _mcq(0), _mcq(0)
    quiz = Quiz(title="T", document_name="d.pdf", questions=[a, b])
    pattern = [True, True, False, False, False]
    same = [_copy((a, "0" if ok else "1", ok, float(ok)), (b, "0" if ok else "1", ok, float(ok)))
            for ok in pattern]
    opposite = [_copy((a, "0" if ok else "1", ok, float(ok)), (b, "1" if ok else "0", not ok, float(not ok)))
                for ok in pattern]

    assert analytics.item_analysis(quiz, same)["reliability"] == 1.0
    assert analytics.item_analysis(quiz, opposite)["reliability"] is None   # tous les totaux égaux


def test_gradebook_analyses_first_attempts_only():
    question = _mcq(0)
    quiz = Quiz(title="T", document_name="d.pdf", questions=[question])
    results = [{"quiz_id": quiz.id, "student_username": "lina", "student_name": "Lina",
                "submitted_at": f"2026-10-01T09:{minute:02d}:00Z", "percentage": 100.0 * ok,
                "graded_answers": [{"question_id": question.id, "student_answer": "0" if ok else "",
                                    "correct": ok, "score": float(ok)}]}
               for minute, ok in ((10, False), (20, True))]
    analysis = analytics.gradebook(quiz, results)["analysis"]
    assert (analysis["students"], analysis["items"][0]["success_rate"]) == (1, 0.0)
