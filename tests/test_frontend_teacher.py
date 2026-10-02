"""Espace enseignant de l'interface Streamlit, exécuté avec AppTest : les
appels HTTP sont remplacés par des doublures qui les enregistrent."""
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

DOC = {"id": "d1", "filename": "cours.pdf", "num_chunks": 3, "uploaded_by": "prof",
       "uploaded_at": "2026-09-30T00:00:00Z"}
QUIZZES = [
    {"id": qid, "title": f"Quiz {qid}", "document_name": "cours.pdf",
     "created_at": "2026-09-30T00:00:00Z", "published": False, "created_by": "prof",
     "agent_report": None,
     "questions": [{"id": f"{qid}-1", "type": "qcm", "theme": "", "difficulty": "moyen",
                    "question": "Question ?", "choices": ["A", "B", "C", "D"],
                    "correct_choice_index": 1, "reference_answer": "B", "explanation": "",
                    "source_excerpt": ""}]}
    for qid in ("qa", "qb")
]
MAP = {"document_id": "d1", "method": "acp", "num_points": 1,
       "points": [{"x": 0.5, "y": 0.5, "preview": "texte", "page": 1, "chunk_index": 0}]}


class _Response:
    def __init__(self, payload=None, content=b"", status=200, url=""):
        self.status_code, self.ok, self._payload, self.url = status, status < 400, payload, url
        self.content = content or json.dumps(payload).encode()
        self.text = self.content.decode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} Server Error for url: {self.url}")


def _fake_get(calls, quizzes=QUIZZES, fail=()):
    def get(url, **kwargs):
        path = urlsplit(url).path
        calls.append(path)
        if path in fail:
            return _Response({"detail": "Internal Server Error"}, status=500, url=url)
        if path == "/health":
            return _Response({"status": "ok", "llm_provider": "mock"})
        if path == "/documents":
            return _Response([DOC])
        if path == "/quizzes":
            return _Response(quizzes)
        if path == "/documents/d1/map":
            return _Response(MAP)
        if path.endswith("/export/pdf"):
            return _Response(content=b"%PDF-1.4 test")
        if path.endswith("/export/json"):
            return _Response(content=b"{}")
        raise AssertionError(f"appel inattendu : GET {url}")
    return get


NEW_QUIZ = {**QUIZZES[0], "id": "qn", "title": "Quiz généré"}


def _fake_post(posts, quizzes=None):
    def post(url, **kwargs):
        path = urlsplit(url).path
        posts.append(path)
        if path == "/quizzes/generate":
            return _Response(NEW_QUIZ)
        for quiz in quizzes or []:
            if path == f"/quizzes/{quiz['id']}/publish":
                quiz["published"] = True
        return _Response({**QUIZZES[0], "published": True})
    return post


def _teacher_app():
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "jeton"
    at.session_state["user"] = {"username": "prof", "role": "professeur"}
    return at


def _exports(calls):
    return [c for c in calls if "/export/" in c]


ALL_EXPORTS = sorted(f"/quizzes/{q}/export/{k}" for q in ("qa", "qb") for k in ("pdf", "json"))


def test_exports_are_fetched_once_and_shared_across_reruns_and_logins():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls)):
        at.run()
        assert not at.exception
        assert sorted(_exports(calls)) == ALL_EXPORTS
        assert sorted(d.proto.label for d in at.get("download_button")) == \
            ["Télécharger le JSON"] * 2 + ["Télécharger le PDF"] * 2

        at.run()
        # Autre session, autre jeton : la clé de cache ne contient pas le jeton.
        other = AppTest.from_file(APP, default_timeout=30)
        other.session_state["token"] = "autre-jeton"
        other.session_state["user"] = {"username": "prof2", "role": "professeur"}
        other.run()
        assert not other.exception
    assert sorted(_exports(calls)) == ALL_EXPORTS


def test_publishing_refetches_that_quizs_exports_only():
    """L'export JSON contient le drapeau « published » : il fait partie de la clé."""
    calls, posts, quizzes = [], [], json.loads(json.dumps(QUIZZES))
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls, quizzes)), \
         patch("requests.post", side_effect=_fake_post(posts, quizzes)):
        at.run()
        next(b for b in at.button if b.key == "pub_qa").click()
        at.run()
        assert not at.exception
    assert sorted(_exports(calls)) == sorted(ALL_EXPORTS + ["/quizzes/qa/export/json", "/quizzes/qa/export/pdf"])


def test_a_failing_export_shows_its_error_on_that_quiz_only():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls, fail={"/quizzes/qa/export/pdf"})):
        at.run()
    assert not at.exception
    qa, qb = (next(e for e in at.expander if e.label.startswith(f"Quiz {q}")) for q in ("qa", "qb"))
    assert [e.value for e in qa.error] == [
        "Export PDF indisponible : 500 Server Error for url: http://localhost:8000/quizzes/qa/export/pdf"]
    assert [d.proto.label for d in qa.get("download_button")] == ["Télécharger le JSON"]
    assert list(qb.error) == []
    assert sorted(d.proto.label for d in qb.get("download_button")) == ["Télécharger le JSON", "Télécharger le PDF"]


def test_reruns_reuse_cached_reads_until_a_write():
    calls, posts = [], []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls)), \
         patch("requests.post", side_effect=_fake_post(posts)):
        at.run()
        at.run()
        assert not at.exception
        reads = ("/documents", "/quizzes", "/documents/d1/map")
        assert [calls.count(p) for p in reads] == [1, 1, 1]

        next(b for b in at.button if b.key == "pub_qa").click()
        at.run()
        assert not at.exception
    assert posts == ["/quizzes/qa/publish"]
    assert calls.count("/quizzes") == 2


def test_writes_rerun_the_whole_space_and_keep_their_success_message():
    calls, posts = [], []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls)), \
         patch("requests.post", side_effect=_fake_post(posts)):
        at.run()
        next(b for b in at.button if b.label == "Générer le quiz").click()
        at.run()
        assert not at.exception
        titles = [m.value for m in at.markdown if "qb-section__title" in m.value]
        assert any('qb-section__title">Quiz généré<' in t for t in titles)
        assert calls.count("/quizzes") == 2

        next(b for b in at.button if b.label == "Publier pour les étudiants").click()
        at.run()
        assert not at.exception
    assert posts == ["/quizzes/generate", "/quizzes/qn/publish"]
    assert [s.value for s in at.success] == ["Quiz publié."]
