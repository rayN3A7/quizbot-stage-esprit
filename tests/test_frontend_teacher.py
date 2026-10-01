"""Espace enseignant de l'interface Streamlit, exécuté avec AppTest : les
appels HTTP sont remplacés par des doublures qui les enregistrent."""
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

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
    def __init__(self, payload=None, content=b""):
        self.status_code, self.ok, self._payload = 200, True, payload
        self.content = content or json.dumps(payload).encode()
        self.text = self.content.decode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _fake_get(calls):
    def get(url, **kwargs):
        path = urlsplit(url).path
        calls.append(path)
        if path == "/health":
            return _Response({"status": "ok", "llm_provider": "mock"})
        if path == "/documents":
            return _Response([DOC])
        if path == "/quizzes":
            return _Response(QUIZZES)
        if path == "/documents/d1/map":
            return _Response(MAP)
        if path.endswith("/export/pdf"):
            return _Response(content=b"%PDF-1.4 test")
        if path.endswith("/export/json"):
            return _Response(content=b"{}")
        raise AssertionError(f"appel inattendu : GET {url}")
    return get


def _teacher_app():
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "jeton"
    at.session_state["user"] = {"username": "prof", "role": "professeur"}
    return at


def _exports(calls):
    return [c for c in calls if "/export/" in c]


def test_exports_are_fetched_only_when_requested():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls)):
        at.run()
        at.run()
        assert not at.exception
        assert _exports(calls) == []

        next(b for b in at.button if b.key == "prepare_pdf_qa").click()
        at.run()
        assert not at.exception
        assert _exports(calls) == ["/quizzes/qa/export/pdf"]
        assert [d.proto.label for d in at.get("download_button")] == ["Télécharger le PDF"]

        at.run()
    assert _exports(calls) == ["/quizzes/qa/export/pdf"]
