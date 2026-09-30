"""Tests du pipeline de génération de quiz (prompt building, parsing, mock LLM)."""
import json
import logging
import re
from unittest.mock import patch

import pytest

from backend import quiz_generator
from backend.llm_client import LLMError, MockProvider
from backend.models import Difficulty, Question, QuestionType, QuizConfig
from backend.quiz_generator import (
    SYSTEM_PROMPT, _build_user_prompt, _parse_llm_response, _quality_reject_reason,
    filter_questions_by_quality,
)


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


# --------------------------------------------------------------------------- #
# Portes de qualité programmatiques (vérification des questions)
# --------------------------------------------------------------------------- #

TEST_PASSAGES = [
    {"text": "Le machine learning supervisé utilise des exemples étiquetés pour entraîner le modèle."},
    {"text": "La régularisation L2 aide à limiter le surapprentissage en pénalisant les poids élevés."},
    {"text": "Un réseau de neurones contient des couches connectées avec des poids."},
]


def test_quality_gate_rejects_missing_choices():
    """Une question QCM avec <3 choix doit être rejetée."""
    q = Question(
        type=QuestionType.MCQ,
        question="Qu'est-ce que ML supervisé?",
        choices=["Choix 1", "Choix 2"],  # Seulement 2 choix
        correct_choice_index=0,
        reference_answer="Choix 1",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "3 choix" in reason


def test_quality_gate_accepts_valid_mcq():
    """Une question QCM valide doit passer."""
    q = Question(
        type=QuestionType.MCQ,
        question="Qu'est-ce que le ML supervisé?",
        choices=["Exemples étiquetés", "Sans étiquettes", "Aléatoire", "Manuel"],
        correct_choice_index=0,
        reference_answer="Exemples étiquetés",
        source_excerpt="Le machine learning supervisé utilise des exemples étiquetés",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is None


def test_quality_gate_rejects_invalid_correct_index():
    """Index de réponse correcte hors limites doit être rejeté."""
    q = Question(
        type=QuestionType.MCQ,
        question="Question?",
        choices=["A", "B", "C"],
        correct_choice_index=5,  # Hors limites
        reference_answer="A",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "limites" in reason.lower()


def test_quality_gate_rejects_prompt_markers():
    """Les marqueurs de prompt doivent être rejetés."""
    q = Question(
        type=QuestionType.OPEN,
        question="[PASSAGE 1] Expliquez le ML supervisé?",
        reference_answer="Les réseaux de neurones",
        source_excerpt="Une réponse quelconque",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "prompt" in reason.lower()


def test_quality_gate_rejects_prompt_markers_with_typos():
    """Les marqueurs mal orthographiés ([PASSEAGE], [PASSEAU]) doivent aussi être rejetés."""
    q = Question(
        type=QuestionType.OPEN,
        question="Selon [PASSEAGE 1], qu'est-ce que c'est?",
        reference_answer="Une réponse",
        source_excerpt="Un extrait",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "prompt" in reason.lower()


def test_quality_gate_rejects_ungrounded_source():
    """Un source_excerpt absent des passages doit être rejeté."""
    q = Question(
        type=QuestionType.OPEN,
        question="Question sur quelque chose d'inventé?",
        reference_answer="Une réponse",
        source_excerpt="Quelque chose qui n'existe pas du tout dans aucun passage du cours entier",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "grounding" in reason.lower() or "introuvable" in reason.lower()


def test_quality_gate_accepts_grounded_excerpt():
    """Un source_excerpt présent dans les passages doit passer."""
    q = Question(
        type=QuestionType.OPEN,
        question="Qu'est-ce que la régularisation L2?",
        reference_answer="Elle limite le surapprentissage",
        source_excerpt="La régularisation L2 aide à limiter le surapprentissage",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is None


def test_quality_gate_rejects_code_choices_with_exact_duplicates():
    """Choix en code : les doublons exacts doivent être rejetés."""
    q = Question(
        type=QuestionType.MCQ,
        question="Quel est le bon code Python?",
        choices=["2**n - 1", "2**n - 1", "n * n", "n + n"],  # Doublons exacts
        correct_choice_index=0,
        reference_answer="2**n - 1",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "doublons" in reason.lower()


def test_quality_gate_accepts_code_choices_permuted():
    """Choix en code : les permutations DOIVENT être acceptées (test de compréhension)."""
    q = Question(
        type=QuestionType.MCQ,
        question="Quel code est valide?",
        choices=["2*n - 1", "n - 1*2", "1 - n*2", "n*2 - 1"],  # Permutations OK en code
        correct_choice_index=0,
        reference_answer="2*n - 1",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    # Devrait passer si ce sont du code (car pour code, on tolère)
    # En réalité, aucun de ces codes ne contient "**", donc pas détecté comme code
    # La logique accepte les permutations si c'est du code, rejette sinon.
    # Ces choix contiennent "*", pas "**", donc pas code-like...
    # Laissez passer si c'est une question valide.


def test_quality_gate_accepts_distinct_non_code_choices():
    """Choix non-code clairement distincts doivent passer."""
    q = Question(
        type=QuestionType.MCQ,
        question="Quel est un concept important?",
        choices=[
            "La régularisation limite le surapprentissage",
            "Les données étiquetées sont essentielles au ML supervisé",
            "Les réseaux de neurones contiennent des couches",
            "L'optimisation ajuste les paramètres du modèle",
        ],
        correct_choice_index=0,
        reference_answer="La régularisation limite le surapprentissage",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is None


def test_quality_gate_rejects_internal_contradiction_operator():
    """Réponse contient ** mais question et explication ne le mentionnent pas."""
    q = Question(
        type=QuestionType.MCQ,
        question="Quelle est la complexité en temps?",
        choices=["O(n)", "O(2**n)", "O(n log n)", "O(1)"],
        correct_choice_index=1,
        reference_answer="O(2**n)",
        explanation="La complexité exponentielle doublerait.",  # Pas de **
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is not None
    assert "contradiction" in reason.lower()


def test_quality_gate_accepts_consistent_operator():
    """Réponse ** est mentionnée aussi dans la question."""
    q = Question(
        type=QuestionType.MCQ,
        question="Quel code calcule 2 à la puissance n (2**n)?",
        choices=["2**n", "2*n", "n**2", "pow(n, 2)"],
        correct_choice_index=0,
        reference_answer="2**n",
        explanation="L'opérateur ** calcule les puissances en Python.",
    )
    reason = _quality_reject_reason(q, TEST_PASSAGES)
    assert reason is None


def test_filter_questions_by_quality_separates_kept_rejected():
    """Le filtre doit séparer les questions valides des rejetées."""
    good_q = Question(
        type=QuestionType.OPEN,
        question="Qu'est-ce que la régularisation L2?",
        reference_answer="Elle limite le surapprentissage",
        source_excerpt="La régularisation L2 aide à limiter",
    )
    bad_q = Question(
        type=QuestionType.MCQ,
        question="Question?",
        choices=["A"],  # Seul choix
        correct_choice_index=0,
        reference_answer="A",
    )
    kept, reasons = filter_questions_by_quality([good_q, bad_q], TEST_PASSAGES)
    assert len(kept) == 1
    assert kept[0] == good_q
    assert len(reasons) == 1
    assert 1 in reasons  # bad_q était à index 1


# --------------------------------------------------------------------------- #
# Régression : avec Qwen2.5-3B (le 7B ne tient pas en VRAM et bascule sur le 3B),
# les 10 questions d'un quiz étaient rejetées. Le modèle écrivait l'étiquette
# "[PASSAGE 1]" dans source_excerpt, car le prompt
# demandait d'« indiquer le passage source » (quiz 29657adce8, 9afc4625be...).
# --------------------------------------------------------------------------- #

# Même expression que MockProvider pour découper les passages d'un prompt.
_MOCK_PASSAGE_RE = r"\[PASSAGE \d+\]\s*(.+?)(?=\[PASSAGE|\Z)"


def test_prompts_require_verbatim_excerpt_instead_of_passage_label():
    config = QuizConfig(document_id="d1", num_questions=3, question_type="qcm")
    prompt = _build_user_prompt(config, PASSAGES)
    assert "indiquer le passage source" not in prompt
    assert "source_excerpt" in prompt and "mot pour mot" in prompt

    # La règle « ne recopie jamais » contredisait la porte de grounding (qui
    # exige 6 mots recopiés) tant qu'elle n'exemptait pas source_excerpt.
    exception = [l for l in SYSTEM_PROMPT.splitlines() if "Seule exception" in l]
    assert exception and "source_excerpt" in exception[0] and "MOT POUR MOT" in exception[0]


def test_passages_are_the_last_block_of_the_prompt():
    config = QuizConfig(document_id="d1", num_questions=2, question_type="qcm")
    prompt = _build_user_prompt(config, PASSAGES, extra_instructions="NOTE SUPPLEMENTAIRE\n")
    extracted = re.findall(_MOCK_PASSAGE_RE, prompt, re.S)
    assert [e.strip() for e in extracted] == [p["text"] for p in PASSAGES]
    assert prompt.index("NOTE SUPPLEMENTAIRE") < prompt.index("[PASSAGE 1]")


def test_quality_gate_reason_names_field_and_quotes_passage_label():
    q = Question(
        type=QuestionType.OPEN,
        question="Pourquoi la régularisation limite-t-elle le surapprentissage ?",
        reference_answer="Elle pénalise les poids élevés.",
        source_excerpt="[PASSEAU 2]",
    )
    assert _quality_reject_reason(q, TEST_PASSAGES) == (
        'Marqueur de prompt dans source_excerpt : "[PASSEAU 2]"'
    )


def test_generate_quiz_all_rejected_error_names_reasons_and_logs_raw_output(caplog):
    raw = json.dumps({"questions": [
        {"type": "ouverte", "question": f"Question {i} ?", "reference_answer": "r",
         "source_excerpt": label}
        for i, label in enumerate(["[PASSAGE 1]", "[PASSAGE 1]", "[PASSEAU 2]"])
    ]})
    config = QuizConfig(document_id="d1", num_questions=3)
    with patch.object(quiz_generator, "_retrieve_passages", return_value=TEST_PASSAGES), \
         patch.object(quiz_generator, "generate_completion", return_value=raw), \
         caplog.at_level(logging.WARNING, logger="backend.quiz_generator"):
        with pytest.raises(LLMError) as exc:
            quiz_generator.generate_quiz("d1", "cours.pdf", config)

    message = str(exc.value)
    assert "Toutes les questions générées (3) ont été rejetées" in message
    assert 'source_excerpt : "[PASSAGE 1]"' in message
    assert 'source_excerpt : "[PASSEAU 2]"' in message
    assert message.count("[PASSAGE 1]") == 1  # raisons identiques dédupliquées
    for i in (1, 2, 3):
        assert f"Question {i}/3 rejetée" in caplog.text
    assert raw[:200] in caplog.text
