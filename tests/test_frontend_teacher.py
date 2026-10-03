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


def _fake_get(calls, quizzes=QUIZZES, fail=(), docs=(DOC,), books=None):
    def get(url, **kwargs):
        path = urlsplit(url).path
        calls.append(path)
        if path in fail:
            return _Response({"detail": "Internal Server Error"}, status=500, url=url)
        if path == "/health":
            return _Response({"status": "ok", "llm_provider": "mock"})
        if path == "/documents":
            return _Response(list(docs))
        if path == "/quizzes":
            return _Response(quizzes)
        if path == "/documents/d1/map":
            return _Response(MAP)
        if path.startswith("/quizzes/") and path.endswith("/results"):
            quiz_id = path.split("/")[2]
            return _Response((books or {}).get(quiz_id) or _empty_book(quiz_id))
        if path.endswith("/export/pdf"):
            return _Response(content=b"%PDF-1.4 test")
        if path.endswith("/export/json"):
            return _Response(content=b"{}")
        raise AssertionError(f"appel inattendu : GET {url}")
    return get


def _empty_book(quiz_id):
    """Réponse du backend pour un quiz publié sans copie."""
    return {"quiz_id": quiz_id, "title": f"Quiz {quiz_id}", "questions": [], "entries": [],
            "summary": {"students": 0, "mean": None, "median": None, "min": None, "max": None,
                        "passed": 0, "retries": 0}}


NEW_QUIZ = {**QUIZZES[0], "id": "qn", "title": "Quiz généré"}


def _fake_post(posts, quizzes=None, generated=NEW_QUIZ):
    def post(url, **kwargs):
        path = urlsplit(url).path
        posts.append(path)
        if path == "/quizzes/generate":
            return _Response(generated)
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


# --------------------------------------------------------------------------- #
# Groupes rendus d'un seul bloc (entrée échelonnée par --i)
# --------------------------------------------------------------------------- #

def _blocks_with(at, marker):
    return [m.value for m in at.markdown if marker in m.value]


def test_document_library_is_one_staggered_block_with_escaped_names():
    docs = [{**DOC, "id": f"d{k}", "filename": name}
            for k, name in enumerate(["cours.pdf", "<img src=x onerror=alert(1)>.pdf", "tp.pptx"])]
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get([], docs=docs)):
        at.run()
    assert not at.exception
    library = _blocks_with(at, 'class="qb-row"')
    assert len(library) == 1
    assert library[0].count('class="qb-row"') == 3
    # Enfants directs d'un même .qb-stack : le CSS les décale par :nth-child.
    assert library[0].startswith('<div class="qb-stack"><div class="qb-row">')
    assert "&lt;img src=x onerror=alert(1)&gt;.pdf" in library[0] and "<img" not in library[0]


def test_review_cards_are_one_block_and_blank_lines_stay_inside_it():
    generated = {**NEW_QUIZ, "questions": [
        {**QUIZZES[0]["questions"][0], "id": f"g{k}", "question": text}
        for k, text in enumerate(["Première ?", "Deuxième ligne 1\n\nligne 2 ?", "Troisième ?"])
    ]}
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get([])), \
         patch("requests.post", side_effect=_fake_post([], generated=generated)):
        at.run()
        next(b for b in at.button if b.label == "Générer le quiz").click()
        at.run()
    assert not at.exception
    review = _blocks_with(at, 'class="qb-q"')
    assert len(review) == 1
    assert review[0].count('<article class="qb-q"') == 3
    assert review[0].startswith('<div class="qb-stack"><article class="qb-q">')
    assert "\n" not in review[0]
    assert "Deuxième ligne 1<br><br>ligne 2 ?" in review[0]


# --------------------------------------------------------------------------- #
# Onglet Résultats : carnet de notes
# --------------------------------------------------------------------------- #

def _graded(answer, score):
    return {"question_id": "qa-1", "question": "Question ?", "student_answer": answer, "correct": score == 1.0,
            "score": score, "correct_answer": "B", "explanation": ""}


BOOK = {
    "quiz_id": "qa", "title": "Quiz qa", "questions": [{"id": "qa-1", "type": "qcm", "question": "Question ?"}],
    "summary": {"students": 2, "mean": 50.0, "median": 50.0, "min": 0.0, "max": 100.0, "passed": 1, "retries": 1},
    "entries": [
        {"student_username": "lina", "student_name": "Lina", "attempt": 1, "counted": True,
         "submitted_at": "2026-10-01T09:10:00Z", "total_score": 1.0, "max_score": 1.0, "percentage": 100.0,
         "graded_answers": [_graded("1", 1.0)]},
        {"student_username": "lina", "student_name": "Lina", "attempt": 2, "counted": False,
         "submitted_at": "2026-10-01T09:20:00Z", "total_score": 0.0, "max_score": 1.0, "percentage": 0.0,
         "graded_answers": [_graded("", 0.0)]},
        {"student_username": "omar", "student_name": "Omar", "attempt": 1, "counted": True,
         "submitted_at": "2026-10-01T09:30:00Z", "total_score": 0.0, "max_score": 1.0, "percentage": 0.0,
         "graded_answers": [_graded("3", 0.0)]},
    ],
}
PUBLISHED = [{**QUIZZES[0], "published": True}, QUIZZES[1]]


def test_results_tab_waits_for_a_published_quiz_without_calling_the_api():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls)):
        at.run()
    assert not at.exception
    assert not [c for c in calls if c.endswith("/results")]
    assert any("Aucun quiz publié" in m.value for m in at.markdown)


def test_results_tab_shows_the_gradebook_and_a_readable_copy():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls, quizzes=PUBLISHED, books={"qa": BOOK})):
        at.run()
        assert not at.exception
        # Seul le quiz publié est proposé : une seule lecture du carnet.
        assert [c for c in calls if c.endswith("/results")] == ["/quizzes/qa/results"]
        tiles = next(m.value for m in at.markdown if "au-dessus de 50 %" in m.value)
        for value in ('qb-stat__n">2<', 'qb-stat__n">50,0 %<', 'qb-stat__n">1/2<'):
            assert value in tiles

        table = at.dataframe[0].value
        assert list(table["Étudiant"]) == ["Lina", "Lina", "Omar"]
        assert list(table["Comptée"]) == [True, False, True]
        assert list(table["Points"]) == ["1 / 1", "0 / 1", "0 / 1"]

        at.selectbox(key="copy_qa").set_value(2)
        at.run()
    assert not at.exception
    copy = next(m.value for m in at.markdown if 'class="qb-res' in m.value)
    # Le QCM est enregistré par index ; l'enseignant lit le choix tel que l'étudiant l'a vu.
    assert "D. D" in copy and "Réponse de l&#x27;étudiant" in copy and "0 / 1" in copy


def test_refresh_rereads_the_gradebook_that_reruns_otherwise_keep_in_cache():
    calls = []
    at = _teacher_app()
    with patch("requests.get", side_effect=_fake_get(calls, quizzes=PUBLISHED, books={"qa": BOOK})):
        at.run()
        at.run()
        assert calls.count("/quizzes/qa/results") == 1
        next(b for b in at.button if b.key == "results_refresh").click()
        at.run()
    assert not at.exception
    assert calls.count("/quizzes/qa/results") == 2


# --------------------------------------------------------------------------- #
# Chargeur de génération : il ne doit jamais rester affiché
# --------------------------------------------------------------------------- #

def test_generation_loader_never_lingers_after_success_or_failure():
    def failing_post(url, **kwargs):
        return _Response({"detail": "Erreur lors de la génération par le LLM : délai dépassé"},
                         status=502, url=url)

    for post, expect_error in ((_fake_post([]), False), (failing_post, True)):
        at = _teacher_app()
        with patch("requests.get", side_effect=_fake_get([])), patch("requests.post", side_effect=post):
            at.run()
            next(b for b in at.button if b.label == "Générer le quiz").click()
            at.run()
        assert not at.exception
        # L'attribut, pas le nom de classe : la feuille de style contient « .qb-loader ».
        assert not [m for m in at.markdown if 'class="qb-loader"' in m.value]
        errors = [e.value for e in at.error]
        assert errors == (["La génération a échoué : Erreur lors de la génération par le LLM : délai dépassé"]
                          if expect_error else [])
