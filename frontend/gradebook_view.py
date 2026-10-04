"""
Mise en forme du carnet de notes (espace enseignant).

Fonctions pures, sans Streamlit : testables directement. Les nombres et les
dates suivent les usages français, comme le reste de l'interface.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime


def num(value) -> str:
    """Nombre à la française, sans zéros inutiles : 0,413 · 28,6 · 4."""
    return f"{round(float(value), 3):g}".replace(".", ",")


def pct(value) -> str:
    return "—" if value is None else f"{value:.1f}".replace(".", ",") + " %"


def rate(value) -> str:
    """Taux entre 0 et 1, en pourcentage entier : 0,6364 -> « 64 % »."""
    return "—" if value is None else f"{100 * value:.0f} %"


def index(value) -> str:
    """Indice entre -1 et 1 (discrimination, fidélité), deux décimales."""
    return "—" if value is None else f"{value:.2f}".replace(".", ",").replace("-", "−")


def reliability_label(alpha) -> str:
    """Seuils usuels de l'alpha de Cronbach."""
    if alpha is None:
        return "non calculée"
    return "bonne" if alpha >= 0.8 else ("acceptable" if alpha >= 0.7 else "faible")


def when(value) -> str:
    """Date de remise, stockée en UTC, affichée à l'heure de cette machine."""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value or "")
    return moment.astimezone().strftime("%d/%m/%Y %H:%M")


def answer_text(answer, question: dict | None) -> str:
    """Réponse telle que l'étudiant l'a vue : un QCM est enregistré par l'index du choix."""
    answer = str(answer)
    question = question or {}
    choices = question.get("choices") or []
    if question.get("type") == "qcm" and answer.strip().isdigit() and int(answer) < len(choices):
        k = int(answer)
        return f"{chr(65 + k)}. {choices[k]}"
    return answer


def _cell(value) -> str:
    """Un nom commençant par = + - @ deviendrait une formule à l'ouverture dans un
    tableur (injection CSV) : on le préfixe d'une apostrophe."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def gradebook_csv(book: dict) -> bytes:
    """Carnet pour un tableur réglé en français : séparateur « ; », virgule décimale,
    UTF-8 avec BOM (sans lui, Excel affiche mal les accents)."""
    question_ids = [q["id"] for q in book["questions"]]
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";", lineterminator="\r\n")
    writer.writerow(["Étudiant", "Identifiant", "Tentative", "Comptée", "Remise le", "Points", "Sur",
                     "Pourcentage"] + [f"Q{k}" for k in range(1, len(question_ids) + 1)])
    for e in book["entries"]:
        scores = {g["question_id"]: g["score"] for g in e["graded_answers"]}
        writer.writerow(
            [_cell(e["student_name"]), _cell(e["student_username"]), e["attempt"],
             "oui" if e["counted"] else "non", when(e["submitted_at"]), num(e["total_score"]),
             num(e["max_score"]), num(e["percentage"])]
            + [num(scores[q]) if q in scores else "" for q in question_ids]
        )
    return out.getvalue().encode("utf-8-sig")
