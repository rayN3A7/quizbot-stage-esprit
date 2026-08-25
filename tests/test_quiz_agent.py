"""
Tests pour l'agent de génération de quiz (quiz_agent.py).

Ces tests forcent LLM_PROVIDER=mock afin de ne nécessiter aucune clé API ni
GPU. Le vérificateur factice de MockProvider contrôle si le texte de la
réponse indiquée apparaît dans les passages fournis : on peut donc tester le
comportement réel (accepter une bonne réponse, rejeter une réponse fabriquée)
sans appel réseau ni chargement de modèle.

Lancer avec :
    pytest tests/test_quiz_agent.py -v
"""
import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest
from unittest.mock import patch

from backend.models import Question, QuestionType, QuizConfig
from backend import quiz_agent


FAKE_PASSAGES = [
    {
        "text": (
            "Le machine learning supervisé apprend une fonction à partir d'exemples "
            "étiquetés. La régression logistique est un algorithme de classification "
            "linéaire couramment utilisé."
        ),
        "metadata": {"chunk_index": 0, "source_page": 1},
        "distance": 0.1,
    },
    {
        "text": (
            "Le surapprentissage (overfitting) survient quand un modèle mémorise le "
            "bruit des données d'entraînement au lieu d'apprendre des motifs "
            "généraux. La régularisation L2 aide à le limiter."
        ),
        "metadata": {"chunk_index": 1, "source_page": 2},
        "distance": 0.2,
    },
]


def test_verify_question_accepts_correct_answer():
    question = Question(
        type=QuestionType.OPEN,
        question="Que fait la régularisation L2 ?",
        reference_answer="La régularisation L2 aide à le limiter",
    )
    result = quiz_agent.verify_question(question, FAKE_PASSAGES)
    assert result.valide is True


def test_verify_question_rejects_fabricated_answer():
    question = Question(
        type=QuestionType.OPEN,
        question="Que fait la régularisation L2 ?",
        reference_answer="La régularisation L2 double automatiquement la taille du dataset",
    )
    result = quiz_agent.verify_question(question, FAKE_PASSAGES)
    assert result.valide is False


def test_generate_quiz_agentic_end_to_end():
    config = QuizConfig(
        document_id="fake_doc",
        title="Quiz de test",
        num_questions=3,
        question_type="mélange",
        use_verification_agent=True,
    )

    with patch.object(quiz_agent, "_retrieve_passages", return_value=FAKE_PASSAGES):
        quiz = quiz_agent.generate_quiz_agentic("fake_doc", "cours_ml.pdf", config)

    assert len(quiz.questions) <= 3
    assert len(quiz.questions) > 0
    assert quiz.agent_report is not None
    assert quiz.agent_report["dropped"] + len(quiz.questions) >= quiz.agent_report["generated"]


def test_generate_quiz_agentic_raises_if_no_document_indexed():
    config = QuizConfig(document_id="empty_doc", num_questions=3, use_verification_agent=True)
    with patch.object(quiz_agent, "_retrieve_passages", side_effect=ValueError("Aucun passage indexé")):
        with pytest.raises(ValueError):
            quiz_agent.generate_quiz_agentic("empty_doc", "vide.pdf", config)