"""Tentatives d'un étudiant : seule la première compte dans les statistiques
agrégées ; les suivantes sont enregistrées et marquées."""
import copy
from datetime import datetime, timezone
from uuid import uuid4

from backend import storage
from backend.config import settings
from backend.models import Question, QuestionType, Quiz, QuizResult


def _result(quiz_id, minute, username, name, pct):
    return {"quiz_id": quiz_id, "student_username": username, "student_name": name,
            "submitted_at": f"2026-10-01T09:{minute:02d}:00Z", "percentage": pct}


def test_first_attempts_keeps_earliest_submission_per_student_and_quiz():
    results = [  # volontairement dans le désordre
        _result("q1", 30, "alice", "Alice", 100.0),   # reprise d'Alice
        _result("q1", 10, "alice", "Alice", 20.0),    # première tentative d'Alice
        _result("q1", 20, "bob", "Bob", 50.0),
        _result("q2", 40, "alice", "Alice", 70.0),    # autre quiz : compte aussi
        # Ancien format, sans student_username : regroupé par nom affiché.
        {"quiz_id": "q1", "student_name": "Carol", "submitted_at": "2026-10-01T09:50:00Z", "percentage": 90.0},
        {"quiz_id": "q1", "student_name": "Carol", "submitted_at": "2026-10-01T09:05:00Z", "percentage": 10.0},
    ]
    kept = sorted((r["quiz_id"], r.get("student_username") or r["student_name"], r["percentage"])
                  for r in storage.first_attempts(results))
    assert kept == [("q1", "Carol", 10.0), ("q1", "alice", 20.0), ("q1", "bob", 50.0), ("q2", "alice", 70.0)]


def test_save_result_numbers_attempts_and_never_overwrites_within_a_second():
    quiz_id = uuid4().hex[:10]
    t = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)
    for microsecond, pct in ((100_000, 0.0), (300_000, 100.0)):  # même seconde arrondie
        storage.save_result(QuizResult(
            quiz_id=quiz_id, student_name="Alice", student_username="alice",
            submitted_at=t.replace(microsecond=microsecond), graded_answers=[],
            total_score=pct / 100, max_score=1, percentage=pct,
        ))

    assert sorted((r["attempt"], r["percentage"]) for r in storage.list_results(quiz_id)) == \
        [(1, 0.0), (2, 100.0)]
    assert [r["percentage"] for r in storage.list_results(quiz_id, first_attempts_only=True)] == [0.0]


def test_old_results_take_their_excerpt_from_the_quiz_when_it_can_be_read():
    quiz = Quiz(title="T", document_name="d.pdf", questions=[
        Question(type=QuestionType.OPEN, question="Q1 ?", reference_answer="r",
                 source_excerpt="Extrait un recopie du cours"),
        Question(type=QuestionType.OPEN, question="Q2 ?", reference_answer="r",
                 source_excerpt="Extrait deux recopie du cours"),
    ])
    storage.save_quiz(quiz)
    corrupt_id = uuid4().hex[:10]
    (settings.QUIZZES_DIR / f"{corrupt_id}.json").write_text("{ pas du json", encoding="utf-8")
    q1, q2 = quiz.questions

    old = {"quiz_id": quiz.id, "student_name": "Ancien", "graded_answers": [
        {"question_id": q1.id, "question": "Q1 ?", "score": 1.0},
        {"question_id": "question-disparue", "question": "Q3 ?", "score": 0.0},
    ]}
    new = {"quiz_id": quiz.id, "graded_answers": [
        {"question_id": q2.id, "question": "Q2 ?", "score": 1.0, "source_excerpt": ""},
    ]}
    deleted_quiz = {"quiz_id": "quiz-supprime", "graded_answers": [{"question_id": "x", "question": "Q ?"}]}
    corrupt_quiz = {"quiz_id": corrupt_id, "graded_answers": [{"question_id": "x", "question": "Q ?"}]}
    originals = copy.deepcopy([old, new, deleted_quiz, corrupt_quiz])

    enriched = storage.with_source_excerpts([old, new, deleted_quiz, corrupt_quiz])

    assert [g.get("source_excerpt") for g in enriched[0]["graded_answers"]] == \
        ["Extrait un recopie du cours", None]
    assert enriched[1:] == originals[1:]             # champ présent, ou quiz illisible : inchangés
    assert [old, new, deleted_quiz, corrupt_quiz] == originals   # rien n'est modifié en place
