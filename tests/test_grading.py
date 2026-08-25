"""Tests du module de correction (grading)."""
from unittest.mock import patch

from backend.models import Difficulty, Question, QuestionType, Quiz, StudentAnswer


def _mcq_question():
    return Question(
        type=QuestionType.MCQ, theme="Reseaux", difficulty=Difficulty.MEDIUM,
        question="Qu'est-ce qu'un neurone artificiel ?",
        choices=["A", "B", "Bonne reponse", "D"],
        correct_choice_index=2,
        reference_answer="Bonne reponse",
        explanation="explication",
    )


def _open_question():
    return Question(
        type=QuestionType.OPEN, theme="Gradient", difficulty=Difficulty.MEDIUM,
        question="Expliquez la retropropagation.",
        reference_answer="La retropropagation ajuste les poids en propageant l'erreur",
        explanation="explication",
    )


def test_grade_mcq_correct_by_index():
    from backend.grading import grade_answer
    q = _mcq_question()
    result = grade_answer(q, "2")
    assert result.correct is True
    assert result.score == 1.0


def test_grade_mcq_incorrect_by_index():
    from backend.grading import grade_answer
    q = _mcq_question()
    result = grade_answer(q, "0")
    assert result.correct is False
    assert result.score == 0.0


def test_grade_open_question_similarity(fake_embed_text):
    from backend.grading import grade_answer
    q = _open_question()
    with patch("backend.grading.embed_text", side_effect=fake_embed_text):
        result_good = grade_answer(q, "La retropropagation ajuste les poids en propageant l'erreur du reseau")
        result_bad = grade_answer(q, "Un sujet totalement different sans rapport")
    assert result_good.score > result_bad.score


def test_grade_quiz_computes_stats(fake_embed_text):
    from backend.grading import grade_quiz
    mcq, open_q = _mcq_question(), _open_question()
    quiz = Quiz(title="T", document_name="doc.pdf", questions=[mcq, open_q])
    answers = [
        StudentAnswer(question_id=mcq.id, answer="2"),
        StudentAnswer(question_id=open_q.id, answer="La retropropagation ajuste les poids en propageant l'erreur"),
    ]
    with patch("backend.grading.embed_text", side_effect=fake_embed_text):
        result = grade_quiz(quiz, "Rayen", answers)

    assert result.max_score == 2.0
    assert 0 <= result.percentage <= 100
    assert "Reseaux" in result.stats_by_theme
    assert "Gradient" in result.stats_by_theme
    assert "qcm" in result.stats_by_type
    assert "ouverte" in result.stats_by_type


def test_grade_quiz_missing_answer_scores_zero():
    from backend.grading import grade_quiz
    mcq = _mcq_question()
    quiz = Quiz(title="T", document_name="doc.pdf", questions=[mcq])
    result = grade_quiz(quiz, "Rayen", answers=[])
    assert result.total_score == 0.0
