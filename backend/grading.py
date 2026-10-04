"""
Module de correction : compare les réponses de l'étudiant aux réponses de
référence. Les QCM sont corrigés par comparaison d'index.

Questions ouvertes : barème à trois paliers, comme un correcteur humain —
juste (1 point), partiel (0,5), faux (0). La similarité sémantique avec la
réponse attendue ne tranche que les cas nets (limite documentée dans le cahier
des charges, section 9 : elle ne capture pas toutes les nuances) :
- réponse vide, ou qui n'ajoute rien à l'énoncé : faux, sans calcul ;
- question sans réponse attendue (défaut de génération) : l'agent de
  correction juge d'après le passage du cours s'il existe ; sinon note
  provisoire à confirmer par l'enseignant ;
- similarité sous OPEN_ANSWER_LOW : faux ; au-dessus de OPEN_ANSWER_HIGH : juste ;
- entre les deux : l'agent de correction tranche (grading_agent.py) ; à
  défaut, partiel provisoire, à confirmer.
Une copie qui s'adresse au correcteur n'est jamais soumise à l'agent : notée
à la seule similarité, que du texte ne peut pas manipuler, et signalée.
Auparavant le score valait la similarité elle-même : « je ne sais pas » ou une
phrase hors sujet rapportaient de 0,1 à 0,4 point par question.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from .config import settings
from .embeddings import cosine_similarity, embed_text
from .grading_agent import OpenAnswerJudge
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


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _fold(text))


# Copie qui s'adresse au correcteur : formule d'adresse, demande d'ignorer les
# consignes, réclamation de la note. Volontairement étroit (« correcteur
# orthographique » ou « ignorer les règles » sont du vocabulaire de cours) : un
# faux positif ne coûte d'ailleurs que l'aide de l'agent, la copie restant notée.
_GRADER_ADDRESS_RE = re.compile(
    r"\b(?:au|aux|pour (?:le|la|les)|cher|chere|monsieur le|madame la)\s+"
    r"(?:correct|evaluat|examinat)(?:eur|rice)s?\b"
    r"|\b(?:ignor|oubli|disregard)\w*\b.{0,40}\b(?:consignes?|instructions?|bareme)\b"
    r"|\bnote\s+(?:maximale?|parfaite)\b|\btous\s+les\s+points\b|\b20\s*/\s*20\b"
    r"|\b(?:full|maximum)\s+(?:marks|points|score)\b|\b(?:to the|dear)\s+(?:grader|examiner)\b"
)


def addresses_the_grader(answer: str) -> bool:
    return bool(_GRADER_ADDRESS_RE.search(_fold(answer)))


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


# (question, copie) -> (verdict, justification), ou None si l'agent ne tranche pas.
Judge = Callable[[Question, str], Optional[tuple[str, str]]]


def _by_agent(judge: Optional[Judge], question: Question, student_answer: str,
              similarity: Optional[float] = None) -> Optional[GradedAnswer]:
    decision = judge(question, student_answer.strip()) if judge else None
    if decision is None:
        return None
    verdict, justification = decision
    return _graded_open(question, student_answer, verdict, "agent",
                        justification or "Corrigée par l'agent de correction.", similarity)


def _grade_open_core(question: Question, student_answer: str, judge: Optional[Judge]) -> GradedAnswer:
    answer = student_answer.strip()
    if not question.reference_answer.strip():
        return _by_agent(judge, question, student_answer) or _graded_open(
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
    return _by_agent(judge, question, student_answer, similarity) or _graded_open(
        question, student_answer, "partiel", "similarité",
        "Votre réponse rejoint en partie la réponse attendue : note provisoire, "
        "à confirmer par l'enseignant.", similarity, needs_review=True,
    )


def _grade_open(question: Question, student_answer: str, judge: Optional[Judge] = None) -> GradedAnswer:
    if not student_answer.strip():
        return _graded_open(question, student_answer, "faux", "règle", "Pas de réponse.")
    if addresses_the_grader(student_answer):
        graded = _grade_open_core(question, student_answer, judge=None)
        graded.needs_review = True
        graded.feedback = ("Votre copie contient des consignes adressées au correcteur : elle a été "
                           "notée sans l'agent de correction et sera vérifiée par l'enseignant.")
        return graded
    return _grade_open_core(question, student_answer, judge)


def grade_answer(question: Question, student_answer: str, judge: Optional[Judge] = None) -> GradedAnswer:
    if question.type == QuestionType.MCQ:
        return _grade_mcq(question, student_answer)
    return _grade_open(question, student_answer, judge)


def grade_quiz(quiz: Quiz, student_name: str, answers: List[StudentAnswer]) -> QuizResult:
    answers_by_id: Dict[str, str] = {a.question_id: a.answer for a in answers}
    questions_by_id: Dict[str, Question] = {q.id: q for q in quiz.questions}

    graded: List[GradedAnswer] = []
    theme_scores = defaultdict(lambda: {"score": 0.0, "max": 0.0})
    type_scores = defaultdict(lambda: {"score": 0.0, "max": 0.0})

    judge = OpenAnswerJudge()  # un par copie : plafonne les appels au modèle
    for question in quiz.questions:
        student_answer = answers_by_id.get(question.id, "")
        result = grade_answer(question, student_answer, judge)
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
