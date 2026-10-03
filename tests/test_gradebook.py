"""Carnet de notes enseignant : copies, tentatives et statistiques de la classe."""
from backend import analytics
from backend.models import Question, QuestionType, Quiz

QUIZ = Quiz(title="Réseaux", document_name="cours.pdf", questions=[
    Question(type=QuestionType.MCQ, question="Q1 ?", choices=["A", "B", "C", "D"],
             correct_choice_index=1, reference_answer="B"),
    Question(type=QuestionType.OPEN, question="Q2 ?", reference_answer="Réponse"),
])


def _copy(minute, pct, username=None, name=""):
    result = {"quiz_id": QUIZ.id, "student_name": name, "submitted_at": f"2026-10-01T09:{minute:02d}:00Z",
              "total_score": pct / 50, "max_score": 2.0, "percentage": pct, "graded_answers": []}
    if username is not None:
        result["student_username"] = username
    return result


def test_attempts_are_numbered_by_date_and_only_first_attempts_count():
    results = [  # volontairement dans le désordre
        _copy(30, 100.0, "alice", "Alice"),   # reprise d'Alice
        _copy(10, 50.0, "alice", "Alice"),
        _copy(20, 0.0, "bob", "Bob"),
        # Ancien format, sans student_username ni attempt : regroupé par nom affiché.
        _copy(50, 100.0, name="Carol"),
        _copy(5, 25.0, name="Carol"),
    ]
    book = analytics.gradebook(QUIZ, results)

    assert [(e["student_name"], e["attempt"], e["counted"], e["percentage"]) for e in book["entries"]] == [
        ("Alice", 1, True, 50.0), ("Alice", 2, False, 100.0),
        ("Bob", 1, True, 0.0),
        ("Carol", 1, True, 25.0), ("Carol", 2, False, 100.0),
    ]
    # Moyenne de 50, 0 et 25 : les reprises à 100 % n'entrent pas dans les statistiques.
    assert book["summary"] == {"students": 3, "mean": 25.0, "median": 25.0, "min": 0.0, "max": 50.0,
                               "passed": 1, "retries": 2}
    assert [q["id"] for q in book["questions"]] == [q.id for q in QUIZ.questions]


def test_two_accounts_sharing_a_display_name_stay_separate():
    book = analytics.gradebook(QUIZ, [_copy(10, 0.0, "sami1", "Sami"), _copy(20, 100.0, "sami2", "Sami")])
    assert [(e["student_username"], e["attempt"]) for e in book["entries"]] == [("sami1", 1), ("sami2", 1)]
    assert book["summary"]["students"] == 2


def test_a_quiz_without_submissions_has_an_empty_summary():
    book = analytics.gradebook(QUIZ, [])
    assert book["entries"] == []
    assert book["summary"] == {"students": 0, "mean": None, "median": None, "min": None, "max": None,
                               "passed": 0, "retries": 0}
