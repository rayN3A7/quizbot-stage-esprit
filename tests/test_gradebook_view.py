"""Mise en forme du carnet de notes : nombres à la française, réponses QCM
lisibles, export CSV pour un tableur réglé en français."""
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))
from gradebook_view import (  # noqa: E402
    answer_text, gradebook_csv, index, num, pct, rate, reliability_label, when,
)

MCQ = {"id": "q1", "type": "qcm", "question": "Q1 ?", "choices": ["Vrai", "Faux", "Peut-être"]}
OPEN = {"id": "q2", "type": "ouverte", "question": "Q2 ?"}


def test_numbers_and_percentages_are_written_the_french_way():
    assert [num(0.413), num(4.0), num(28.6), num(0.0)] == ["0,413", "4", "28,6", "0"]
    assert [pct(57.14), pct(100.0), pct(None)] == ["57,1 %", "100,0 %", "—"]
    assert [rate(0.6364), rate(1.0), rate(None)] == ["64 %", "100 %", "—"]
    assert [index(0.32), index(-0.4), index(None)] == ["0,32", "−0,40", "—"]


def test_reliability_uses_the_usual_cronbach_thresholds():
    assert [reliability_label(a) for a in (0.85, 0.8, 0.72, 0.69, None)] == \
        ["bonne", "bonne", "acceptable", "faible", "non calculée"]


def test_mcq_answers_show_the_choice_the_student_saw():
    assert answer_text("1", MCQ) == "B. Faux"
    assert answer_text("7", MCQ) == "7"            # index hors des choix : laissé tel quel
    assert answer_text("42", OPEN) == "42"         # question ouverte : jamais traduit
    assert answer_text("", MCQ) == ""
    assert answer_text("0", None) == "0"           # question disparue du quiz


def test_submission_time_is_shown_in_local_time():
    expected = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M")
    assert when("2026-10-01T09:30:00Z") == expected
    assert when("pas une date") == "pas une date"


def test_csv_opens_in_a_french_spreadsheet_and_cannot_inject_formulas():
    book = {"questions": [MCQ, OPEN], "entries": [
        {"student_name": "=HYPERLINK(\"http://x\")", "student_username": "lina", "attempt": 1, "counted": True,
         "submitted_at": "2026-10-01T09:30:00Z", "total_score": 1.41, "max_score": 2.0, "percentage": 70.5,
         "graded_answers": [{"question_id": "q1", "score": 1.0}, {"question_id": "q2", "score": 0.41}]},
        {"student_name": "Lina", "student_username": "lina", "attempt": 2, "counted": False,
         "submitted_at": "2026-10-01T10:00:00Z", "total_score": 0.0, "max_score": 2.0, "percentage": 0.0,
         "graded_answers": [{"question_id": "q1", "score": 0.0}]},   # ancienne copie incomplète
    ]}
    data = gradebook_csv(book)

    assert data.startswith("﻿".encode("utf-8"))     # BOM : accents corrects dans Excel
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter=";"))
    assert rows[0] == ["Étudiant", "Identifiant", "Tentative", "Comptée", "Remise le", "Points", "Sur",
                       "Pourcentage", "Q1", "Q2"]
    assert rows[1][0] == "'=HYPERLINK(\"http://x\")"
    assert rows[1][2:4] + rows[1][5:] == ["1", "oui", "1,41", "2", "70,5", "1", "0,41"]
    assert rows[2][2:4] + rows[2][5:] == ["2", "non", "0", "2", "0", "0", ""]
