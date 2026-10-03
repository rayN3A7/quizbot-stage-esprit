"""
Test d'intégration bout-en-bout de l'API FastAPI, authentification comprise.

La base vectorielle (ChromaDB + sentence-transformers) et le modèle
d'embedding sont simulés (mock) afin que ce test s'exécute rapidement et
sans dépendance lourde (torch) ; le LLM utilise le fournisseur "mock" natif
de QuizBot, ce qui teste réellement le pipeline RAG "retrieval -> prompt ->
génération -> parsing -> stockage -> correction -> export" ainsi que le
contrôle d'accès par rôle (professeur / étudiant).
"""
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeVectorStore:
    """Simule ChromaDB : stocke les chunks en mémoire et renvoie les k premiers."""

    def __init__(self):
        self.data = {}

    def index_document(self, document_id, chunks):
        self.data[document_id] = chunks
        return len(chunks)

    def query(self, document_id, query_text, top_k=6):
        chunks = self.data.get(document_id, [])
        return [{"text": c.text, "metadata": {}, "distance": 0.1} for c in chunks[:top_k]]

    def all_texts(self, document_id, limit=400):
        return [c.text for c in self.data.get(document_id, [])[:limit]]

    def raw_chunks(self, document_id, limit=300):
        chunks = self.data.get(document_id, [])[:limit]
        return {"documents": [c.text for c in chunks],
                "metadatas": [{"chunk_index": c.chunk_index, "source_page": c.source_page} for c in chunks],
                "embeddings": []}


def _fake_embed_text(text: str):
    vec = np.zeros(64)
    for w in text.lower().split():
        vec[hash(w) % 64] += 1
    return vec.tolist()


@pytest.fixture
def client():
    fake_store = FakeVectorStore()
    with patch("backend.main.get_vector_store", return_value=fake_store), \
         patch("backend.quiz_generator.get_vector_store", return_value=fake_store), \
         patch("backend.semantic_map.get_vector_store", return_value=fake_store), \
         patch("backend.grading.embed_text", side_effect=_fake_embed_text):
        from backend.main import app
        yield TestClient(app)


def _auth_headers(client, username, role, password="secret123", full_name=""):
    """Enregistre (si besoin) et connecte un utilisateur, retourne ses headers Bearer."""
    client.post("/auth/register", json={
        "username": username, "password": password, "full_name": full_name, "role": role,
    })
    r = client.post("/auth/login", data={"username": username, "password": password})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_and_login(client):
    r = client.post("/auth/register", json={
        "username": "prof_test", "password": "secret123", "full_name": "Prof Test", "role": "professeur",
    })
    assert r.status_code == 201
    assert r.json()["role"] == "professeur"
    assert "hashed_password" not in r.json()

    r = client.post("/auth/login", data={"username": "prof_test", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["role"] == "professeur"

    r = client.post("/auth/login", data={"username": "prof_test", "password": "wrong"})
    assert r.status_code == 401


def test_duplicate_username_rejected(client):
    client.post("/auth/register", json={"username": "dup", "password": "secret123", "role": "etudiant"})
    r = client.post("/auth/register", json={"username": "dup", "password": "secret123", "role": "professeur"})
    assert r.status_code == 400


def test_short_password_rejected(client):
    r = client.post("/auth/register", json={"username": "shortpw", "password": "123", "role": "etudiant"})
    assert r.status_code == 400


def test_endpoints_require_authentication(client):
    r = client.get("/documents")
    assert r.status_code == 401
    r = client.get("/quizzes")
    assert r.status_code == 401


def test_student_cannot_access_professor_endpoints(client):
    stud_headers = _auth_headers(client, "stud_only", "etudiant")
    r = client.get("/documents", headers=stud_headers)
    assert r.status_code == 403
    r = client.post("/quizzes/generate", headers=stud_headers, json={"document_id": "x", "num_questions": 3})
    assert r.status_code == 403


def test_full_teacher_and_student_flow(client, sample_pdf):
    prof_headers = _auth_headers(client, "prof_flow", "professeur", full_name="Prof Flow")
    stud_headers = _auth_headers(client, "stud_flow", "etudiant", full_name="Student Flow")

    # 1. Téléversement (professeur uniquement)
    with open(sample_pdf, "rb") as f:
        r = client.post("/documents/upload", headers=prof_headers,
                         files={"file": ("cours.pdf", f, "application/pdf")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["num_chunks"] >= 1
    assert doc["uploaded_by"] == "prof_flow"

    # 2. Génération du quiz (RAG complet avec LLM mock)
    r = client.post("/quizzes/generate", headers=prof_headers, json={
        "document_id": doc["id"], "title": "Quiz Auto", "num_questions": 4,
        "question_type": "mélange", "difficulty": "moyen",
    })
    assert r.status_code == 200
    quiz = r.json()
    assert len(quiz["questions"]) == 4
    assert quiz["published"] is False
    assert quiz["created_by"] == "prof_flow"

    # 3. Un quiz non publié n'est pas visible/soumissible pour les étudiants
    r = client.get(f"/quizzes/{quiz['id']}", headers=stud_headers)
    assert r.status_code == 403
    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=stud_headers, json={
        "quiz_id": quiz["id"], "answers": [],
    })
    assert r.status_code == 403

    # 4. Publication (professeur uniquement)
    r = client.post(f"/quizzes/{quiz['id']}/publish", headers=stud_headers)
    assert r.status_code == 403
    r = client.post(f"/quizzes/{quiz['id']}/publish", headers=prof_headers)
    assert r.status_code == 200
    assert r.json()["published"] is True

    # 5. Soumission des réponses (toutes correctes) — le nom vient du compte connecté
    answers = []
    for q in quiz["questions"]:
        if q["type"] == "qcm":
            answers.append({"question_id": q["id"], "answer": str(q["correct_choice_index"])})
        else:
            answers.append({"question_id": q["id"], "answer": q["reference_answer"]})

    # Un professeur ne peut pas soumettre de réponses (rôle étudiant requis)
    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=prof_headers, json={
        "quiz_id": quiz["id"], "answers": answers,
    })
    assert r.status_code == 403

    r = client.post(f"/quizzes/{quiz['id']}/submit", headers=stud_headers, json={
        "quiz_id": quiz["id"], "answers": answers,
    })
    assert r.status_code == 200
    result = r.json()
    assert result["percentage"] == 100.0
    assert result["student_name"] == "Student Flow"

    # 6. Exports (professeur uniquement)
    r = client.get(f"/quizzes/{quiz['id']}/export/pdf", headers=stud_headers)
    assert r.status_code == 403
    r = client.get(f"/quizzes/{quiz['id']}/export/pdf", headers=prof_headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"

    r = client.get(f"/quizzes/{quiz['id']}/export/json", headers=prof_headers)
    assert r.status_code == 200


def test_student_token_cannot_retrieve_answers(client, sample_pdf):
    """Un étudiant authentifié qui appelle l'API directement ne doit lire aucune
    réponse avant de soumettre ; le professeur garde tout (relecture, exports)."""
    prof = _auth_headers(client, "prof_leak", "professeur")
    stud = _auth_headers(client, "stud_leak", "etudiant")
    with open(sample_pdf, "rb") as f:
        doc = client.post("/documents/upload", headers=prof,
                          files={"file": ("cours.pdf", f, "application/pdf")}).json()
    quiz = client.post("/quizzes/generate", headers=prof, json={
        "document_id": doc["id"], "num_questions": 4, "question_type": "mélange",
    }).json()
    assert client.post(f"/quizzes/{quiz['id']}/publish", headers=prof).status_code == 200
    expected_answers = [q["correct_choice_index"] for q in quiz["questions"]]
    assert any(i is not None for i in expected_answers)

    hidden = {"correct_choice_index", "reference_answer", "explanation", "source_excerpt"}
    as_student = [client.get(f"/quizzes/{quiz['id']}", headers=stud).json()]
    as_student += [q for q in client.get("/quizzes", headers=stud).json() if q["id"] == quiz["id"]]
    assert len(as_student) == 2
    for payload in as_student:
        assert [q["question"] for q in payload["questions"]] == [q["question"] for q in quiz["questions"]]
        for q in payload["questions"]:
            assert hidden.isdisjoint(q), f"exposé à l'étudiant : {sorted(hidden & q.keys())}"
            if q["type"] == "qcm":
                assert q["choices"]

    as_professor = client.get(f"/quizzes/{quiz['id']}", headers=prof).json()
    assert [q["correct_choice_index"] for q in as_professor["questions"]] == expected_answers
    listed = next(q for q in client.get("/quizzes", headers=prof).json() if q["id"] == quiz["id"])
    assert [q["correct_choice_index"] for q in listed["questions"]] == expected_answers
    exported = client.get(f"/quizzes/{quiz['id']}/export/json", headers=prof).json()
    assert [q["correct_choice_index"] for q in exported["questions"]] == expected_answers


def test_only_first_attempt_counts_in_the_map_overlay(client, sample_pdf):
    """Refaire un quiz est permis (auto-évaluation) : la reprise est enregistrée
    et marquée, mais seule la première tentative entre dans la carte."""
    from backend import storage

    prof = _auth_headers(client, "prof_attempts", "professeur")
    alice = _auth_headers(client, "alice_attempts", "etudiant")
    bob = _auth_headers(client, "bob_attempts", "etudiant")
    with open(sample_pdf, "rb") as f:
        doc = client.post("/documents/upload", headers=prof,
                          files={"file": ("cours.pdf", f, "application/pdf")}).json()
    quiz = client.post("/quizzes/generate", headers=prof, json={
        "document_id": doc["id"], "num_questions": 4, "question_type": "mélange",
    }).json()
    client.post(f"/quizzes/{quiz['id']}/publish", headers=prof)

    questions = quiz["questions"]
    blank = [{"question_id": q["id"], "answer": ""} for q in questions]
    right = [{"question_id": q["id"],
              "answer": str(q["correct_choice_index"]) if q["type"] == "qcm" else q["reference_answer"]}
             for q in questions]

    def submit(headers, answers):
        r = client.post(f"/quizzes/{quiz['id']}/submit", headers=headers,
                        json={"quiz_id": quiz["id"], "answers": answers})
        assert r.status_code == 200
        return r.json()

    # Carte à un seul point (le cours de test tient en un chunk, index 0) : chaque
    # réponse y est rattachée par son extrait ; l'aperçu reprend les énoncés pour
    # les éventuels extraits trop courts, rattachés à l'ancienne.
    preview = " ".join(q["question"] for q in questions)

    def overlay():
        one_point = {"document_id": doc["id"], "method": "acp", "num_points": 1,
                     "points": [{"x": 0.5, "y": 0.5, "page": 1, "chunk_index": 0, "preview": preview}]}
        with patch("backend.main.build_semantic_map", return_value=one_point):
            r = client.get(f"/documents/{doc['id']}/map", headers=prof,
                           params={"with_performance": True})
        point = r.json()["points"][0]
        return point["attempts"], point["score"]

    first = submit(alice, blank)
    after_first = overlay()
    second = submit(alice, right)
    assert (first["attempt"], second["attempt"]) == (1, 2)
    assert (first["percentage"], second["percentage"]) == (0.0, 100.0)
    assert overlay() == after_first

    saved = [r for r in storage.list_results(quiz["id"]) if r["student_username"] == "alice_attempts"]
    assert sorted(r["attempt"] for r in saved) == [1, 2]

    submit(bob, right)
    assert overlay()[0] == after_first[0] + len(questions)


def test_gradebook_is_for_professors_and_lists_every_attempt(client, sample_pdf):
    prof = _auth_headers(client, "prof_book", "professeur")
    lina = _auth_headers(client, "lina_book", "etudiant", full_name="Lina")
    with open(sample_pdf, "rb") as f:
        doc = client.post("/documents/upload", headers=prof,
                          files={"file": ("cours.pdf", f, "application/pdf")}).json()
    quiz = client.post("/quizzes/generate", headers=prof, json={
        "document_id": doc["id"], "num_questions": 4, "question_type": "mélange",
    }).json()
    client.post(f"/quizzes/{quiz['id']}/publish", headers=prof)

    right = [{"question_id": q["id"],
              "answer": str(q["correct_choice_index"]) if q["type"] == "qcm" else q["reference_answer"]}
             for q in quiz["questions"]]
    blank = [{"question_id": q["id"], "answer": ""} for q in quiz["questions"]]
    for answers in (blank, right):
        r = client.post(f"/quizzes/{quiz['id']}/submit", headers=lina,
                        json={"quiz_id": quiz["id"], "answers": answers})
        assert r.status_code == 200

    # Les copies portent les noms des étudiants : réservées aux professeurs.
    assert client.get(f"/quizzes/{quiz['id']}/results", headers=lina).status_code == 403
    assert client.get(f"/quizzes/{quiz['id']}/results").status_code == 401
    assert client.get("/quizzes/inconnu/results", headers=prof).status_code == 404

    r = client.get(f"/quizzes/{quiz['id']}/results", headers=prof)
    assert r.status_code == 200
    book = r.json()
    assert [(e["student_username"], e["student_name"], e["attempt"], e["counted"], e["percentage"])
            for e in book["entries"]] == [("lina_book", "Lina", 1, True, 0.0), ("lina_book", "Lina", 2, False, 100.0)]
    assert (book["summary"]["students"], book["summary"]["mean"], book["summary"]["retries"]) == (1, 0.0, 1)
    assert len(book["entries"][1]["graded_answers"]) == len(quiz["questions"])


def test_old_result_files_are_attached_through_their_quiz_excerpt(client, sample_pdf):
    """Résultats enregistrés avant GradedAnswer.source_excerpt : l'extrait est
    relu dans le quiz ; si le quiz a disparu, l'appariement d'origine reste."""
    from backend.config import settings

    prof = _auth_headers(client, "prof_backfill", "professeur")
    with open(sample_pdf, "rb") as f:
        doc = client.post("/documents/upload", headers=prof,
                          files={"file": ("cours.pdf", f, "application/pdf")}).json()
    quiz = client.post("/quizzes/generate", headers=prof, json={
        "document_id": doc["id"], "num_questions": 4, "question_type": "mélange",
    }).json()

    def overlay_attempts():
        # Aperçu sans aucun mot commun avec les énoncés : seul l'extrait peut rattacher.
        one_point = {"document_id": doc["id"], "method": "acp", "num_points": 1,
                     "points": [{"x": 0.5, "y": 0.5, "page": 1, "chunk_index": 0, "preview": "zzz"}]}
        with patch("backend.main.build_semantic_map", return_value=one_point):
            r = client.get(f"/documents/{doc['id']}/map", headers=prof,
                           params={"with_performance": True})
        assert r.status_code == 200
        return r.json()["points"][0]["attempts"]

    def write_old_format_result(quiz_id, student_name):
        old = {"quiz_id": quiz_id, "student_name": student_name, "submitted_at": "2026-08-10T23:42:01Z",
               "graded_answers": [{"question_id": q["id"], "question": q["question"], "student_answer": "",
                                   "correct": False, "score": 0.0, "correct_answer": "", "explanation": ""}
                                  for q in quiz["questions"]],
               "total_score": 0.0, "max_score": float(len(quiz["questions"])), "percentage": 0.0}
        results_dir = settings.DATA_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        (results_dir / f"{quiz_id}_{student_name}.json").write_text(json.dumps(old), encoding="utf-8")

    before = overlay_attempts()
    write_old_format_result("quiz-supprime", "Orphelin")
    assert overlay_attempts() == before
    write_old_format_result(quiz["id"], "Ancien")
    assert overlay_attempts() == before + len(quiz["questions"])


def test_generate_quiz_unknown_document_returns_404(client):
    prof_headers = _auth_headers(client, "prof_404", "professeur")
    r = client.post("/quizzes/generate", headers=prof_headers, json={"document_id": "unknown", "num_questions": 3})
    assert r.status_code == 404


def test_upload_rejects_unsupported_format(client, tmp_path):
    prof_headers = _auth_headers(client, "prof_badfmt", "professeur")
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with open(bad_file, "rb") as f:
        r = client.post("/documents/upload", headers=prof_headers,
                         files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 400


def test_professor_signup_code_enforced(client):
    with patch("backend.main.settings.PROFESSOR_SIGNUP_CODE", "letmein"):
        r = client.post("/auth/register", json={
            "username": "prof_needs_code", "password": "secret123", "role": "professeur",
        })
        assert r.status_code == 403

        r = client.post("/auth/register", json={
            "username": "prof_with_code", "password": "secret123", "role": "professeur",
            "professor_code": "letmein",
        })
        assert r.status_code == 201

        # Un étudiant n'a pas besoin du code
        r = client.post("/auth/register", json={
            "username": "student_no_code_needed", "password": "secret123", "role": "etudiant",
        })
        assert r.status_code == 201
