"""Tests du pipeline de génération de quiz (prompt building, parsing, mock LLM)."""
import pytest

from backend.llm_client import MockProvider
from backend.models import Difficulty, QuestionType, QuizConfig
from backend.quiz_generator import SYSTEM_PROMPT, _build_user_prompt, _parse_llm_response


PASSAGES = [
    {"text": "Un reseau de neurones artificiel est un modele de calcul inspire du cerveau."},
    {"text": "La retropropagation ajuste les poids en propageant l'erreur vers l'entree."},
]


def test_build_user_prompt_contains_config_values():
    config = QuizConfig(document_id="d1", num_questions=5, question_type="qcm", difficulty=Difficulty.HARD)
    prompt = _build_user_prompt(config, PASSAGES)
    assert "NUM_QUESTIONS = 5" in prompt
    assert "TYPE_QUESTIONS = qcm" in prompt
    assert "DIFFICULTE = difficile" in prompt
    assert "[PASSAGE 1]" in prompt and "[PASSAGE 2]" in prompt


def test_mock_provider_generates_requested_number_of_questions():
    config = QuizConfig(document_id="d1", num_questions=4, question_type="mélange")
    prompt = _build_user_prompt(config, PASSAGES)
    raw = MockProvider().generate(SYSTEM_PROMPT, prompt)
    questions = _parse_llm_response(raw)
    assert len(questions) == 4


def test_mock_provider_alternates_types_for_melange():
    config = QuizConfig(document_id="d1", num_questions=4, question_type="mélange")
    prompt = _build_user_prompt(config, PASSAGES)
    raw = MockProvider().generate(SYSTEM_PROMPT, prompt)
    questions = _parse_llm_response(raw)
    types_present = {q.type for q in questions}
    assert QuestionType.MCQ in types_present
    assert QuestionType.OPEN in types_present


def test_mock_provider_qcm_only():
    config = QuizConfig(document_id="d1", num_questions=3, question_type="qcm")
    prompt = _build_user_prompt(config, PASSAGES)
    raw = MockProvider().generate(SYSTEM_PROMPT, prompt)
    questions = _parse_llm_response(raw)
    assert all(q.type == QuestionType.MCQ for q in questions)
    for q in questions:
        assert q.choices and len(q.choices) == 4
        assert q.correct_choice_index is not None


def test_parse_llm_response_rejects_invalid_json():
    from backend.llm_client import LLMError
    with pytest.raises(LLMError):
        _parse_llm_response("this is not json")


# --------------------------------------------------------------------------- #
# Régression : les distracteurs QCM du mode mock ne doivent plus être du
# texte générique/absurde ("Un concept sans rapport avec le cours", etc.)
# mais de vraies phrases tirées du document, avec le bon index correct.
# --------------------------------------------------------------------------- #

REALISTIC_PASSAGES = [
    {"text": "Spring Boot is the main project of the Spring Framework and simplifies application development significantly."},
    {"text": "Maven is a build automation tool used primarily for Java projects, managing dependencies via pom.xml."},
    {"text": "ChromaDB is a vector database used to store embeddings for semantic search and retrieval."},
]


def test_mock_provider_mcq_choices_are_real_distinct_sentences():
    config = QuizConfig(document_id="d1", num_questions=3, question_type="qcm")
    prompt = _build_user_prompt(config, REALISTIC_PASSAGES)
    raw = MockProvider().generate(SYSTEM_PROMPT, prompt)
    questions = _parse_llm_response(raw)

    generic_placeholders = {
        "une notion directement liée au passage étudié",
        "un concept sans rapport avec le cours",
        "une erreur typographique du document",
        "une référence à un autre module",
    }
    for q in questions:
        assert q.choices is not None
        assert len(q.choices) == len(set(q.choices)), "les choix doivent être distincts"
        for choice in q.choices:
            assert choice.strip().lower() not in generic_placeholders
        # Le choix marqué correct doit correspondre exactement à reference_answer.
        assert q.choices[q.correct_choice_index] == q.reference_answer


def test_mock_provider_never_leaks_institutional_boilerplate():
    passages_with_boilerplate = [
        {"text": "© 2022-2023 – ESPRIT – Module TEST\nSpring Boot simplifies configuration management for developers."},
        {"text": "© 2022-2023 – ESPRIT – Module TEST\nMaven automates dependency resolution using a central repository."},
    ]
    config = QuizConfig(document_id="d1", num_questions=4, question_type="mélange")
    prompt = _build_user_prompt(config, passages_with_boilerplate)
    raw = MockProvider().generate(SYSTEM_PROMPT, prompt)
    questions = _parse_llm_response(raw)

    for q in questions:
        assert "ESPRIT" not in q.question
        assert "ESPRIT" not in q.reference_answer
        if q.choices:
            assert all("ESPRIT" not in c for c in q.choices)
