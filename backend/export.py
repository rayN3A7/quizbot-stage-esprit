"""
Module d'export : génère un PDF imprimable (ReportLab) ou un JSON structuré
à partir d'un Quiz, pour intégration dans un ENT / LMS.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from .config import settings
from .models import Quiz, QuizResult


def export_quiz_json(quiz: Quiz, include_answers: bool = True) -> Path:
    path = settings.EXPORTS_DIR / f"{quiz.id}.json"
    data = quiz.model_dump(mode="json")
    if not include_answers:
        for q in data["questions"]:
            q.pop("correct_choice_index", None)
            q.pop("reference_answer", None)
            q.pop("explanation", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_result_json(result: QuizResult) -> Path:
    path = settings.EXPORTS_DIR / f"{result.quiz_id}_{result.student_name}_result.json"
    path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def export_quiz_pdf(quiz: Quiz, include_answers: bool = True) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable,
    )

    path = settings.EXPORTS_DIR / f"{quiz.id}.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("QuizTitle", parent=styles["Title"], textColor=colors.HexColor("#C8102E"))
    h2 = ParagraphStyle("QuestionHeader", parent=styles["Heading2"], spaceBefore=14, spaceAfter=4)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)

    # Paragraph interprète un balisage XML : tout contenu (un cours HTML contient
    # <BR>, <TITLE>...) est échappé ; seuls nos <b> et <i> restent du balisage.
    story = [
        Paragraph(escape(quiz.title), title_style),
        Paragraph(f"Document source : {escape(quiz.document_name)}", small),
        Paragraph(f"Généré le : {quiz.created_at.strftime('%d/%m/%Y %H:%M')}", small),
        HRFlowable(width="100%", color=colors.HexColor("#C8102E"), thickness=1, spaceAfter=10, spaceBefore=10),
    ]

    for i, q in enumerate(quiz.questions, start=1):
        story.append(Paragraph(f"Question {i} ({q.type.value.upper()} — {q.difficulty.value})", h2))
        story.append(Paragraph(escape(q.question), body))

        if q.type.value == "qcm" and q.choices:
            items = []
            for idx, choice in enumerate(q.choices):
                label = f"{chr(65 + idx)}. {escape(choice)}"
                if include_answers and idx == q.correct_choice_index:
                    label = f"<b>{label} (bonne réponse)</b>"
                items.append(ListItem(Paragraph(label, body)))
            story.append(ListFlowable(items, bulletType="bullet"))
        elif include_answers:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<i>Réponse attendue :</i> {escape(q.reference_answer)}", body))

        if include_answers and q.explanation:
            story.append(Paragraph(f"<i>Explication :</i> {escape(q.explanation)}", small))
        story.append(Spacer(1, 8))

    doc.build(story)
    return path
