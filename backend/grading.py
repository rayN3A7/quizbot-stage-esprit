"""
Module de correction : compare les réponses de l'étudiant aux réponses de
référence. Les QCM sont corrigés par comparaison d'index.

Questions ouvertes : barème à trois paliers, comme un correcteur humain —
juste (1 point), partiel (0,5), faux (0). La similarité sémantique avec la
réponse attendue ne tranche que les cas nets (limite documentée dans le cahier
des charges, section 9 : elle ne capture pas toutes les nuances) :
- réponse vide, ou qui n'ajoute rien à l'énoncé : faux, sans calcul ;
- question sans réponse attendue (défaut de génération) : rien à comparer,
  note provisoire à confirmer par l'enseignant ;
- similarité sous OPEN_ANSWER_LOW : faux ; au-dessus de OPEN_ANSWER_HIGH : juste ;
- entre les deux : partiel provisoire, à confirmer.
Auparavant le score valait la similarité elle-même : « je ne sais pas » ou une
phrase hors sujet rapportaient de 0,1 à 0,4 point par question.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional

from .config import settings
from .embeddings import cosine_similarity, embed_text
from .models import (
    GradedAnswer,
    Question,
    QuestionType,
    Quiz,
    QuizResult,
    StudentAnswer,
)


def _grade_mcq(question: Question, student_answer: str) -> GradedAnswer:
    correct_index = question.correct_choice_index
    correct_text = (
        question.choices[correct_index]
        if question.choices and correct_index is not None and correct_index < len(question.choices)
        else question.reference_answer
    )

    student_clean = student_answer.strip()
    is_correct = False
    if student_clean.isdigit() and question.choices:
        is_correct = int(student_clean) == correct_index
    else:
        is_correct = student_clean.strip().lower() == correct_text.strip().lower()

    return GradedAnswer(
        question_id=question.id,
        question=question.question,
        student_answer=student_answer,
        correct=is_correct,
        score=1.0 if is_correct else 0.0,
        correct_answer=correct_text,
        explanation=question.explanation,
        source_excerpt=question.source_excerpt,
    )


POINTS = {"juste": 1.0, "partiel": 0.5, "faux": 0.0}

# Mots-outils : ils ne portent pas, à eux seuls, une réponse.
_STOPWORDS = frozenset("""
les des une aux ces cet cette avec comme comment dans donc dont car elle elles est etre ete ont ils
leur leurs lui mais meme mes moi mon nos notre nous par pas pour quand que quel quelle quelles quels
qui quoi sans ses son sont sur tes toi ton vos votre vous cela ceci celle celui ceux alors ainsi
aussi entre chaque tres plus moins peu bien tout tous
""".split())


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", folded)


def _content_words(words: list[str]) -> set[str]:
    return {w for w in words if (len(w) >= 3 or w.isdigit()) and w not in _STOPWORDS}


def adds_nothing(answer: str, question: Question) -> bool:
    """Aucun mot porteur de sens de la réponse n'est absent de l'énoncé : recopier
    la question n'y répond pas. Exception : un énoncé qui CITE sa propre réponse
    (« Expliquez l'idée suivante : « … » », questions du générateur factice) ;
    la citation est retirée de l'énoncé, et la reprendre reste une réponse."""
    stem = f" {' '.join(_words(question.question))} "
    quoted = " ".join(_words(question.reference_answer))
    if quoted and f" {quoted} " in stem:
        stem = stem.replace(f" {quoted} ", " ")
    return not (_content_words(_words(answer)) - _content_words(stem.split()))


def _graded_open(question: Question, student_answer: str, verdict: str, graded_by: str,
                 feedback: str, similarity: Optional[float] = None,
                 needs_review: bool = False) -> GradedAnswer:
    return GradedAnswer(
        question_id=question.id,
        question=question.question,
        student_answer=student_answer,
        correct=verdict == "juste",
        score=POINTS[verdict],
        correct_answer=question.reference_answer,
        explanation=question.explanation,
        source_excerpt=question.source_excerpt,
        feedback=feedback,
        graded_by=graded_by,
        similarity=None if similarity is None else round(similarity, 3),
        needs_review=needs_review,
    )


def _grade_open(question: Question, student_answer: str) -> GradedAnswer:
    answer = student_answer.strip()
    if not answer:
        return _graded_open(question, student_answer, "faux", "règle", "Pas de réponse.")
    if not question.reference_answer.strip():
        return _graded_open(
            question, student_answer, "faux", "règle",
            "Cette question n'a pas de réponse attendue enregistrée : votre réponse "
            "sera corrigée par l'enseignant.", needs_review=True,
        )
    if adds_nothing(answer, question):
        return _graded_open(question, student_answer, "faux", "règle",
                            "La réponse reprend l'énoncé sans rien y ajouter.")

    similarity = max(cosine_similarity(embed_text(answer), embed_text(question.reference_answer)), 0.0)
    if similarity >= settings.OPEN_ANSWER_HIGH:
        return _graded_open(question, student_answer, "juste", "similarité",
                            "Votre réponse rejoint la réponse attendue.", similarity)
    if similarity < settings.OPEN_ANSWER_LOW:
        return _graded_open(question, student_answer, "faux", "similarité",
                            "Votre réponse s'éloigne trop de la réponse attendue.", similarity)
    return _graded_open(
        question, student_answer, "partiel", "similarité",
        "Votre réponse rejoint en partie la réponse attendue : note provisoire, "
        "à confirmer par l'enseignant.", similarity, needs_review=True,
    )


def grade_answer(question: Question, student_answer: str) -> GradedAnswer:
    if question.type == QuestionType.MCQ:
        return _grade_mcq(question, student_answer)
    return _grade_open(question, student_answer)


def grade_quiz(quiz: Quiz, student_name: str, answers: List[StudentAnswer]) -> QuizResult:
    answers_by_id: Dict[str, str] = {a.question_id: a.answer for a in answers}
    questions_by_id: Dict[str, Question] = {q.id: q for q in quiz.questions}

    graded: List[GradedAnswer] = []
    theme_scores = defaultdict(lambda: {"score": 0.0, "max": 0.0})
    type_scores = defaultdict(lambda: {"score": 0.0, "max": 0.0})

    for question in quiz.questions:
        student_answer = answers_by_id.get(question.id, "")
        result = grade_answer(question, student_answer)
        graded.append(result)

        theme_scores[question.theme or "Général"]["score"] += result.score
        theme_scores[question.theme or "Général"]["max"] += 1.0
        type_scores[question.type.value]["score"] += result.score
        type_scores[question.type.value]["max"] += 1.0

    total_score = sum(g.score for g in graded)
    max_score = float(len(quiz.questions)) or 1.0
    _ = questions_by_id  # conservé pour usage futur (traçabilité éventuelle)

    return QuizResult(
        quiz_id=quiz.id,
        student_name=student_name,
        graded_answers=graded,
        total_score=round(total_score, 2),
        max_score=max_score,
        percentage=round(100 * total_score / max_score, 1),
        stats_by_theme={k: round(100 * v["score"] / v["max"], 1) for k, v in theme_scores.items() if v["max"]},
        stats_by_type={k: round(100 * v["score"] / v["max"], 1) for k, v in type_scores.items() if v["max"]},
    )
