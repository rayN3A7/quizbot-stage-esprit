"""Tests du module d'export (PDF / JSON)."""
import json

from backend.export import export_quiz_json, export_quiz_pdf
from backend.models import Question, QuestionType, Quiz


def _quiz():
    q1 = Question(
        type=QuestionType.MCQ, theme="Reseaux", question="Question QCM ?",
        choices=["A", "B", "C", "D"], correct_choice_index=1,
        reference_answer="B", explanation="exp",
    )
    q2 = Question(
        type=QuestionType.OPEN, theme="Gradient", question="Question ouverte ?",
        reference_answer="reponse attendue", explanation="exp2",
    )
    return Quiz(title="Quiz Test Export", document_name="cours.pdf", questions=[q1, q2])


def test_export_quiz_json_includes_answers_by_default():
    quiz = _quiz()
    path = export_quiz_json(quiz, include_answers=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "Quiz Test Export"
    assert data["questions"][0]["correct_choice_index"] == 1


def test_export_quiz_json_can_hide_answers():
    quiz = _quiz()
    path = export_quiz_json(quiz, include_answers=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "correct_choice_index" not in data["questions"][0]
    assert "reference_answer" not in data["questions"][0]


def test_export_quiz_pdf_produces_valid_pdf():
    quiz = _quiz()
    path = export_quiz_pdf(quiz)
    assert path.exists()
    assert path.stat().st_size > 0

    import fitz
    doc = fitz.open(path)
    text = doc[0].get_text()
    assert "Quiz Test Export" in text
    assert "Question QCM" in text
