"""
Suivi des résultats côté enseignant : carnet de notes et analyse des questions.

Toutes les copies remises figurent au carnet, chacune avec son rang de
tentative. Seule la première tentative de chaque étudiant entre dans les
statistiques de la classe et dans l'analyse : les reprises sont de
l'entraînement (même règle que la carte sémantique, voir storage.first_attempts).

Analyse des questions (théorie classique des tests) :
- taux de réussite : part des copies où la question est juste ;
- discrimination : corrélation entre réussir la question et le score obtenu
  sur le RESTE du quiz. Positive, la question est mieux réussie par les élèves
  forts ; négative, par les faibles — signe d'une clé fausse ou d'un énoncé
  ambigu. Seuils d'Ebel : en dessous de 0,20 la question discrimine mal ;
- répartition des choix d'un QCM : un distracteur jamais choisi ne sert à
  rien, un distracteur plus choisi que la bonne réponse est suspect ;
- fidélité du quiz : alpha de Cronbach sur les copies complètes.
Sous MIN_STUDENTS copies, ces indices ne veulent pas dire grand-chose : ils
restent calculés mais aucun signalement statistique n'est émis.
"""
from __future__ import annotations

import statistics
from typing import Optional

from . import storage
from .models import Question, QuestionType, Quiz

PASS_MARK = 50.0  # pourcentage à partir duquel une copie compte comme réussie
MIN_STUDENTS = 5
EASY, HARD = 0.90, 0.25       # au-delà / en deçà de ce taux de réussite, la question est signalée
WEAK_DISCRIMINATION = 0.20
OFTEN_BLANK = 0.30


def _fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _fr_pct(rate: float) -> str:
    return f"{round(100 * rate)} %"


def _choice_of(answer: str, choices: list[str]) -> Optional[int]:
    """Choix d'un QCM : enregistré par index, ou par son texte dans d'anciens fichiers."""
    text = str(answer).strip()
    if text.isdigit() and int(text) < len(choices):
        return int(text)
    lowered = [c.strip().lower() for c in choices]
    return lowered.index(text.lower()) if text.lower() in lowered else None


def _correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < MIN_STUDENTS or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None  # trop peu de copies, ou tout le monde au même niveau : indéfini
    return round(statistics.correlation(xs, ys), 2)


def _cronbach_alpha(matrix: list[list[float]]) -> Optional[float]:
    """Fidélité : les questions mesurent-elles la même chose ? (copies complètes)"""
    if len(matrix) < MIN_STUDENTS or len(matrix[0]) < 2:
        return None
    k = len(matrix[0])
    total_variance = statistics.variance([sum(row) for row in matrix])
    if total_variance == 0:
        return None
    item_variance = sum(statistics.variance(column) for column in zip(*matrix))
    return round(k / (k - 1) * (1 - item_variance / total_variance), 2)


def _choice_counts(question: Question, graded: list[dict]) -> Optional[dict]:
    if question.type != QuestionType.MCQ or not question.choices:
        return None
    counts, other = [0] * len(question.choices), 0
    for g in graded:
        answer = str(g.get("student_answer", ""))
        if not answer.strip():
            continue
        k = _choice_of(answer, question.choices)
        if k is None:
            other += 1
        else:
            counts[k] += 1
    return {
        "options": [{"letter": chr(65 + k), "text": text, "count": counts[k],
                     "correct": k == question.correct_choice_index}
                    for k, text in enumerate(question.choices)],
        "other": other,
    }


def _flags(question: Question, n: int, rate: Optional[float], blank: int,
           discrimination: Optional[float], choices: Optional[dict]) -> list[dict]:
    flags: list[dict] = []

    def flag(level: str, code: str, text: str) -> None:
        flags.append({"level": level, "code": code, "text": text})

    # Défaut de la question elle-même : signalé quel que soit le nombre de copies.
    if question.type == QuestionType.OPEN and not question.reference_answer.strip():
        flag("alerte", "no_reference_answer",
             "Aucune réponse attendue n'est enregistrée : la correction automatique "
             "de cette question n'a pas de sens.")
    if n < MIN_STUDENTS:
        return flags

    if rate >= EASY:
        flag("info", "too_easy", f"Très facile : {_fr_pct(rate)} de réussite.")
    elif rate <= HARD:
        flag("alerte", "too_hard",
             f"Très difficile : {_fr_pct(rate)} de réussite. Vérifiez la bonne réponse et l'énoncé.")
    if discrimination is not None and discrimination < 0:
        flag("alerte", "negative_discrimination",
             "Les élèves qui réussissent le reste du quiz échouent plus souvent ici : "
             "bonne réponse ou énoncé probablement ambigus.")
    elif discrimination is not None and discrimination < WEAK_DISCRIMINATION:
        flag("info", "weak_discrimination",
             f"Distingue mal les élèves forts des élèves faibles (indice {_fr(discrimination)}).")
    if choices:
        key = next((o for o in choices["options"] if o["correct"]), None)
        for option in choices["options"]:
            if option["correct"] or key is None:
                continue
            if option["count"] > key["count"]:
                flag("alerte", "distractor_beats_key",
                     f"Le choix {option['letter']} attire plus d'élèves que la bonne réponse "
                     f"({key['letter']}).")
            elif option["count"] == 0:
                flag("info", "unused_distractor",
                     f"Le choix {option['letter']} n'est jamais choisi : distracteur à remplacer.")
    if blank / n >= OFTEN_BLANK:
        flag("info", "often_blank", f"Laissée sans réponse par {_fr_pct(blank / n)} des élèves.")
    return flags


def item_analysis(quiz: Quiz, copies: list[dict]) -> dict:
    """`copies` : premières tentatives seulement, une par étudiant."""
    question_ids = {q.id for q in quiz.questions}
    answers_by_copy = [{g.get("question_id"): g for g in c.get("graded_answers", [])
                        if g.get("question_id") in question_ids} for c in copies]
    items = []
    for number, question in enumerate(quiz.questions, start=1):
        copies_with_it = [answers for answers in answers_by_copy if question.id in answers]
        graded = [answers[question.id] for answers in copies_with_it]
        n = len(graded)
        correct = [1.0 if g.get("correct") else 0.0 for g in graded]
        rest = [sum(float(g.get("score", 0.0)) for qid, g in answers.items() if qid != question.id)
                for answers in copies_with_it]
        blank = sum(1 for g in graded if not str(g.get("student_answer", "")).strip())
        rate = statistics.fmean(correct) if n else None
        discrimination = _correlation(correct, rest)
        choices = _choice_counts(question, graded)
        items.append({
            "question_id": question.id,
            "number": number,
            "type": question.type.value,
            "question": question.question,
            "n": n,
            "blank": blank,
            "success_rate": None if rate is None else round(rate, 3),
            "mean_score": round(statistics.fmean(float(g.get("score", 0.0)) for g in graded), 3) if n else None,
            "discrimination": discrimination,
            "choices": choices,
            "flags": _flags(question, n, rate, blank, discrimination, choices),
        })

    complete = [[float(answers[q.id].get("score", 0.0)) for q in quiz.questions]
                for answers in answers_by_copy if len(answers) == len(quiz.questions)]
    return {
        "min_students": MIN_STUDENTS,
        "students": len(copies),
        "reliability": _cronbach_alpha(complete) if complete else None,
        "items": items,
    }


def _summary(percentages: list[float]) -> dict:
    def r(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value, 1)

    return {
        "students": len(percentages),
        "mean": r(statistics.fmean(percentages)) if percentages else None,
        "median": r(statistics.median(percentages)) if percentages else None,
        "min": r(min(percentages, default=None)),
        "max": r(max(percentages, default=None)),
        "passed": sum(p >= PASS_MARK for p in percentages),
    }


def gradebook(quiz: Quiz, results: list[dict]) -> dict:
    """`results` : copies brutes de ce quiz, toutes tentatives et anciens formats
    compris (sans student_username ni attempt : regroupées par nom affiché)."""
    entries = [
        {
            "student_username": result.get("student_username") or "",
            "student_name": result.get("student_name") or result.get("student_username") or "",
            "attempt": result["attempt"],
            "counted": result["attempt"] == 1,
            "submitted_at": result.get("submitted_at"),
            "total_score": result.get("total_score", 0.0),
            "max_score": result.get("max_score", float(len(quiz.questions))),
            "percentage": result.get("percentage", 0.0),
            "graded_answers": result.get("graded_answers", []),
        }
        for result in storage.number_attempts(results)
    ]
    entries.sort(key=lambda e: (e["student_name"].casefold(), e["student_username"], e["attempt"]))
    first_copies = [e for e in entries if e["counted"]]
    counted = [e["percentage"] for e in first_copies]
    return {
        "quiz_id": quiz.id,
        "title": quiz.title,
        "questions": [{"id": q.id, "type": q.type.value, "question": q.question} for q in quiz.questions],
        "summary": {**_summary(counted), "retries": len(entries) - len(counted)},
        "entries": entries,
        "analysis": item_analysis(quiz, first_copies),
    }
