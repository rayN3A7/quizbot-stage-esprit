"""
Suivi des résultats côté enseignant : carnet de notes d'un quiz.

Toutes les copies remises y figurent, chacune avec son rang de tentative.
Seule la première tentative de chaque étudiant entre dans les statistiques de
la classe : les reprises sont de l'entraînement (même règle que la carte
sémantique, voir storage.first_attempts).
"""
from __future__ import annotations

import statistics
from typing import Optional

from . import storage
from .models import Quiz

PASS_MARK = 50.0  # pourcentage à partir duquel une copie compte comme réussie


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
    counted = [e["percentage"] for e in entries if e["counted"]]
    return {
        "quiz_id": quiz.id,
        "title": quiz.title,
        "questions": [{"id": q.id, "type": q.type.value, "question": q.question} for q in quiz.questions],
        "summary": {**_summary(counted), "retries": len(entries) - len(counted)},
        "entries": entries,
    }
