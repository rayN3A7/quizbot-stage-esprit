"""Tentatives d'un étudiant : seule la première compte dans les statistiques
agrégées ; les suivantes sont enregistrées et marquées."""
from datetime import datetime, timezone
from uuid import uuid4

from backend import storage
from backend.models import QuizResult


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
