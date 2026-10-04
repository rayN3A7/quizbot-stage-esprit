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


def test_open_answers_show_their_points_their_reason_and_a_provisional_tag():
    graded = [
        {"question_id": "m1", "question": "Question sautée ?", "student_answer": "Les poids changent",
         "correct": False, "score": 0.5, "correct_answer": "B1", "explanation": "",
         "feedback": "Votre réponse rejoint en partie la réponse attendue : note provisoire, à confirmer.",
         "graded_by": "similarité", "similarity": 0.58, "needs_review": True},
        {"question_id": "m2", "question": "Question répondue ?", "student_answer": "x", "correct": True,
         "score": 1.0, "correct_answer": "C2", "explanation": "", "feedback": "Juste : 2 < 3 est bien vu.",
         "graded_by": "agent", "similarity": 0.61, "needs_review": False},
        # Résultat enregistré avant les paliers : ni points ni raison affichés.
        {"question_id": "m3", "question": "Ancienne ?", "student_answer": "y", "correct": False,
         "score": 0.41, "correct_answer": "", "explanation": ""},
    ]
    at, fake_post = _student_app([], result={**RESULT, "graded_answers": graded})
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", side_effect=fake_post):
        at.run()
        next(b for b in at.button if b.label == "Remettre ma copie").click()
        at.run()
    assert not at.exception
    rows = next(m.value for m in at.markdown if 'class="qb-res' in m.value)
    cards = rows.split('<div class="qb-res qb-res--')[1:]

    assert [c.split('"')[0] for c in cards] == ["mid", "ok", "ko"]
    tag = 'class="qb-tag qb-tag--azure">note provisoire</span>'
    assert cards[0].count(tag) == 1 and "0,5 / 1" in cards[0]
    assert "Juste : 2 &lt; 3 est bien vu." in cards[1] and tag not in cards[1]
    assert "Points" not in cards[2] and "— (aucune enregistrée)" in cards[2]
    # Qui a corrigé, et avec quelle similarité : réservé à l'enseignant.
    assert "qb-res__meta" not in rows and "0,58" not in rows


def test_correction_loader_never_lingers_after_success_or_failure():
    def failing_post(url, **kwargs):
        raise RuntimeError("Le serveur ne répond pas")

    for post_factory, expect_error in ((None, False), (failing_post, True)):
        at, fake_post = _student_app([])
        with patch("requests.get", side_effect=_fake_get), \
             patch("requests.post", side_effect=post_factory or fake_post):
            at.run()
            next(b for b in at.button if b.label == "Remettre ma copie").click()
            at.run()
        assert not at.exception
        # L'attribut, pas le nom de classe : la feuille de style contient « .qb-loader ».
        assert not [m for m in at.markdown if 'class="qb-loader"' in m.value]
        assert [e.value for e in at.error] == (["La remise a échoué : Le serveur ne répond pas"]
                                              if expect_error else [])


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
