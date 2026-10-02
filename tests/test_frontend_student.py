"""Espace étudiant de l'interface Streamlit, exécuté avec AppTest (sans
navigateur ni backend : les appels HTTP sont remplacés par des doublures)."""
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

# Forme reçue par un étudiant : aucune bonne réponse dans la charge utile.
QUIZ = {
    "id": "qz1", "title": "Quiz de test", "document_name": "cours.pdf",
    "created_at": "2026-09-30T00:00:00Z", "published": True, "created_by": "prof",
    "questions": [
        {"id": "m1", "type": "qcm", "theme": "", "difficulty": "moyen",
         "question": "Question sautée ?", "choices": ["A1", "B1", "C1", "D1"]},
        {"id": "m2", "type": "qcm", "theme": "", "difficulty": "moyen",
         "question": "Question répondue ?", "choices": ["A2", "B2", "C2", "D2"]},
    ],
}
RESULT = {
    "quiz_id": "qz1", "student_name": "etu", "submitted_at": "2026-09-30T00:00:00Z",
    "graded_answers": [], "total_score": 0.0, "max_score": 2.0, "percentage": 0.0,
    "stats_by_theme": {}, "stats_by_type": {"qcm": 0.0},
}


class _Response:
    def __init__(self, payload):
        self.status_code, self.ok, self._payload = 200, True, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _fake_get(url, **kwargs):
    if url.endswith("/health"):
        return _Response({"status": "ok", "llm_provider": "mock"})
    if url.endswith("/quizzes/qz1"):
        return _Response(QUIZ)
    if url.endswith("/quizzes"):
        return _Response([QUIZ])
    raise AssertionError(f"appel inattendu : GET {url}")


def _student_app(submissions, result=RESULT):
    def fake_post(url, **kwargs):
        assert url.endswith("/quizzes/qz1/submit"), url
        submissions.append(kwargs["json"])
        return _Response(result)

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "jeton"
    at.session_state["user"] = {"username": "etu", "role": "etudiant"}
    return at, fake_post


def test_mcq_choices_are_not_preselected_and_skipped_question_sends_empty():
    submissions = []
    at, fake_post = _student_app(submissions)
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", side_effect=fake_post):
        at.run()
        assert not at.exception
        assert [r.value for r in at.radio] == [None, None]

        at.radio[1].set_value(2)
        next(b for b in at.button if b.label == "Remettre ma copie").click()
        at.run()

    assert not at.exception
    assert submissions == [{
        "quiz_id": "qz1",
        "answers": [
            {"question_id": "m1", "answer": ""},
            {"question_id": "m2", "answer": "2"},
        ],
    }]


def test_result_rows_are_one_block_and_multiline_answers_stay_inside_it():
    """Une réponse ouverte sur plusieurs paragraphes contient une ligne vide :
    dans un bloc HTML, elle ferait sortir la suite de sa carte."""
    graded = [
        {"question_id": "m1", "question": "Question sautée ?", "student_answer": "", "correct": False,
         "score": 0.0, "correct_answer": "B1", "explanation": ""},
        {"question_id": "m2", "question": "Question répondue ?", "student_answer": "ligne 1\n\nligne 2",
         "correct": True, "score": 1.0, "correct_answer": "C2", "explanation": "Parce que."},
    ]
    submissions = []
    at, fake_post = _student_app(submissions, result={**RESULT, "graded_answers": graded})
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", side_effect=fake_post):
        at.run()
        next(b for b in at.button if b.label == "Remettre ma copie").click()
        at.run()
    assert not at.exception
    rows = [m.value for m in at.markdown if 'class="qb-res' in m.value]
    assert len(rows) == 1
    assert rows[0].count('class="qb-res qb-res--') == 2
    assert rows[0].startswith('<div class="qb-stack"><div class="qb-res qb-res--')
    assert "\n" not in rows[0]
    assert "ligne 1<br><br>ligne 2" in rows[0] and "— (vide)" in rows[0]
