"""
Module de correction : compare les réponses de l'étudiant aux réponses de
référence. Les QCM sont corrigés par comparaison d'index ; les questions
ouvertes par similarité sémantique des embeddings (limite documentée dans
le cahier des charges, section 9 : la comparaison sémantique peut ne pas
capturer toutes les nuances des réponses).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

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


def _grade_open(question: Question, student_answer: str) -> GradedAnswer:
    if not student_answer.strip():
        similarity = 0.0
    else:
        student_vec = embed_text(student_answer)
        reference_vec = embed_text(question.reference_answer)
        similarity = max(cosine_similarity(student_vec, reference_vec), 0.0)

    is_correct = similarity >= settings.OPEN_ANSWER_SIMILARITY_THRESHOLD
    return GradedAnswer(
        question_id=question.id,
        question=question.question,
        student_answer=student_answer,
        correct=is_correct,
        score=round(similarity, 3),
        correct_answer=question.reference_answer,
        explanation=question.explanation,
        source_excerpt=question.source_excerpt,
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
