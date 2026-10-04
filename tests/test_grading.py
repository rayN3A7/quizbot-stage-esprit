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


def test_graded_answers_carry_the_question_source_excerpt(fake_embed_text):
    """La carte sémantique rattache chaque réponse au passage via cet extrait."""
    from backend.grading import grade_answer
    mcq, open_q = _mcq_question(), _open_question()
    mcq.source_excerpt = "Un neurone artificiel calcule une somme ponderee de ses entrees"
    open_q.source_excerpt = "La retropropagation ajuste les poids en propageant l'erreur"
    with patch("backend.grading.embed_text", side_effect=fake_embed_text):
        graded = [grade_answer(mcq, "2"), grade_answer(open_q, "texte")]
    assert [g.source_excerpt for g in graded] == [mcq.source_excerpt, open_q.source_excerpt]


# --------------------------------------------------------------------------- #
# Questions ouvertes : paliers juste / partiel / faux
# --------------------------------------------------------------------------- #

def _grade_open_with_similarity(question, answer, similarity):
    """Similarité imposée : on teste le barème, pas le modèle d'embedding."""
    from backend.grading import grade_answer
    with patch("backend.grading.embed_text", return_value=[1.0]) as embed, \
         patch("backend.grading.cosine_similarity", return_value=similarity):
        graded = grade_answer(question, answer)
    return graded, embed.called


def test_open_answers_get_whole_or_half_points_never_their_similarity():
    from backend.config import settings
    q = _open_question()
    below, between, above = settings.OPEN_ANSWER_LOW - 0.01, \
        (settings.OPEN_ANSWER_LOW + settings.OPEN_ANSWER_HIGH) / 2, settings.OPEN_ANSWER_HIGH

    wrong, _ = _grade_open_with_similarity(q, "je ne sais pas", below)
    partial, _ = _grade_open_with_similarity(q, "Les poids changent pendant l'apprentissage", between)
    right, _ = _grade_open_with_similarity(q, "On propage l'erreur pour corriger les poids", above)

    # Avant : une non-réponse rapportait sa similarité (0,1 à 0,4 point).
    assert (wrong.correct, wrong.score, wrong.needs_review) == (False, 0.0, False)
    assert (partial.correct, partial.score, partial.needs_review) == (False, 0.5, True)
    assert (right.correct, right.score, right.needs_review) == (True, 1.0, False)
    assert [g.graded_by for g in (wrong, partial, right)] == ["similarité"] * 3
    assert right.similarity == round(above, 3) and partial.feedback.endswith("à confirmer par l'enseignant.")


def test_copying_the_question_scores_zero_without_comparing_anything():
    q = Question(type=QuestionType.OPEN, question="Quel système expert utilise des règles de production ?",
                 reference_answer="Le système MEDIC")
    pasted, embedded = _grade_open_with_similarity(q, "Quel système expert utilise des règles de production ?",
                                                   similarity=0.99)
    assert (pasted.score, pasted.graded_by, embedded) == (0.0, "règle", False)

    # Reprendre les mots de l'énoncé en ajoutant la réponse reste une réponse.
    answered, embedded = _grade_open_with_similarity(
        q, "Le système expert à règles de production est MEDIC", similarity=0.99)
    assert (answered.score, embedded) == (1.0, True)


def test_a_question_quoting_its_answer_still_accepts_that_answer():
    """Questions du générateur factice : « Expliquez l'idée suivante : « X » », réponse X."""
    q = Question(type=QuestionType.OPEN, reference_answer="La rétropropagation ajuste les poids",
                 question="Expliquez avec vos propres mots l'idée suivante : « La rétropropagation ajuste les poids »")
    graded, embedded = _grade_open_with_similarity(q, "La rétropropagation ajuste les poids", similarity=1.0)
    assert (graded.score, embedded) == (1.0, True)


def test_without_a_reference_answer_the_grade_is_left_to_the_teacher():
    q = Question(type=QuestionType.OPEN, question="Quels défis pose l'acquisition des connaissances ?",
                 reference_answer="")
    graded, embedded = _grade_open_with_similarity(q, "oui", similarity=0.9)
    # Avant : « oui » rapportait 0,447 point, comparé à une réponse attendue vide.
    assert (graded.score, graded.needs_review, embedded) == (0.0, True, False)
    assert "corrigée par l'enseignant" in graded.feedback


def test_blank_open_answer_is_wrong_without_comparison():
    graded, embedded = _grade_open_with_similarity(_open_question(), "   ", similarity=0.9)
    assert (graded.score, graded.feedback, embedded) == (0.0, "Pas de réponse.", False)


def test_results_saved_before_these_fields_still_load():
    from backend.models import GradedAnswer
    old = GradedAnswer(**{"question_id": "q", "question": "Q ?", "student_answer": "x", "correct": False,
                          "score": 0.41, "correct_answer": "r", "explanation": ""})
    assert (old.feedback, old.graded_by, old.similarity, old.needs_review) == ("", "", None, False)


def test_grade_quiz_skipped_mcq_sent_as_empty_string_scores_zero():
    """Le frontend envoie "" pour un QCM sauté (plus de choix A présélectionné)."""
    from backend.grading import grade_quiz
    mcq = _mcq_question()
    quiz = Quiz(title="T", document_name="doc.pdf", questions=[mcq])
    result = grade_quiz(quiz, "Rayen", [StudentAnswer(question_id=mcq.id, answer="")])

    graded = result.graded_answers[0]
    assert graded.correct is False and graded.score == 0.0
    assert result.stats_by_theme == {"Reseaux": 0.0}
