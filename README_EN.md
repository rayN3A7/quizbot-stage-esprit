# QuizBot — Intelligent Teaching Assistant

QuizBot generates self-assessment quizzes (multiple-choice and open questions) from a course
uploaded by a teacher (PDF or PPTX), lets students take them with instant, detailed feedback,
and gives the teacher a gradebook with a per-question analysis.

It is built around a **Retrieval-Augmented Generation (RAG)** pipeline and an open-source
language model (**Qwen2.5-Instruct**) that runs **locally on the GPU** in 4-bit, so no course
content has to leave the machine. OpenAI, Mistral and Hugging Face are also supported through
the same interface.

Summer internship project 2025–2026 (ESPRIT technical specification "Assistant Pédagogique
Intelligent — Quiz Bot"). A French version of this file is available in `README.md`.

---

## Features

**Teacher space**
- Upload a course (PDF or PPTX): text extraction, cleaning of repeated headers/footers,
  chunking (800 tokens, 150 overlap), embedding and indexing in ChromaDB.
- Configure a quiz: title, number of questions (1–20), type (MCQ / open / mixed), difficulty,
  optional themes, optional **verification agent**.
- Review every question with its answer, explanation and the verbatim course excerpt it is
  based on; publish it; export it as **PDF or JSON** (with or without answers).
- **Results tab**: gradebook (one row per submission, class statistics on first attempts),
  CSV export, detailed copies, provisional grades to confirm, and a per-question analysis
  (success rate, discrimination, MCQ choice distribution, Cronbach's alpha, warnings).
- **Course map**: 2D projection of the course (t-SNE / PCA) with the students' results
  overlaid, to see which parts of the course are not understood.

**Student space**
- Take a published quiz; answers and source excerpts are never sent to the student before
  submission.
- Immediate score, correction of every question with an explanation, statistics by theme and
  by question type.

## How it works

```
Course (PDF/PPTX) → extraction → cleaning → chunking → embeddings (all-MiniLM-L6-v2) → ChromaDB
                                                                                       │
Quiz request → document-derived query → top-3k retrieval → MMR (λ = 0.55) → structured prompt
            → LLM (Qwen2.5, 4-bit) → JSON parsing / salvage → quality gates → [verification agent]
            → quiz (JSON store)
```

- **Quality gates** (`backend/quiz_generator.py`): structure, prompt artifacts
  ("[PASSAGE 1]"…), grounding of the cited excerpt (6 consecutive words found in the passages),
  degenerate choices, internal contradictions.
- **Verification agent** (`backend/quiz_agent.py`): generate → verify → regenerate loop; each
  question is checked by the gates, then by an LLM verifier; rejected questions are regenerated
  with the rejection reason (max. 2 retries) or dropped.
- **Grading** (`backend/grading.py`, `backend/grading_agent.py`): MCQs by index; open answers on
  three tiers (right 1 / partial 0.5 / wrong 0). Rules and embedding similarity decide the clear
  cases (< 0.45 wrong, ≥ 0.80 right); a grading agent settles the uncertain band; anything it
  cannot settle stays a provisional grade for the teacher. Copies that address the grader are
  never shown to the agent.

## Project structure

```
backend/
  main.py             FastAPI application and endpoints
  auth.py             bcrypt password hashing, JWT, role-based dependencies
  config.py           central configuration (.env)
  database.py, db_models.py   SQLAlchemy engine and users table (SQLite by default)
  models.py           Pydantic schemas
  ingestion.py        extraction, cleaning, chunking
  embeddings.py       sentence-transformers wrapper
  vectorstore.py      ChromaDB wrapper
  llm_client.py       providers: local, openai, mistral, huggingface, mock
  quiz_generator.py   retrieval, MMR, prompt, parsing, quality gates
  quiz_agent.py       verification agent
  grading.py          MCQ and three-tier open-answer grading
  grading_agent.py    LLM judge for uncertain open answers
  analytics.py        gradebook and item analysis
  semantic_map.py     2D course map and performance overlay
  export.py           PDF (ReportLab) and JSON export
  storage.py          JSON persistence of documents, quizzes and results
frontend/
  app.py              Streamlit application (login, teacher space, student space)
  gradebook_view.py   gradebook formatting helpers (CSV, tables)
  hero.py, semantic_map_view.py
scripts/
  evaluate.py         evaluation of quality gates and retrieval
  evaluate_grading.py open-answer grading benchmark (grading_benchmark.json)
tests/                150 pytest tests (+ a PDF fixture)
docs/USER_GUIDE.md    user guide for teachers (English); GUIDE_UTILISATION.md (French)
```

## Installation

Requirements: Python 3.10+; for the local model, an NVIDIA GPU with CUDA (tested on an
RTX 3060 Laptop GPU, 6 GB VRAM).

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then edit .env
```

For the **local model** (`LLM_PROVIDER=local`), also install PyTorch with CUDA support
(see pytorch.org for the command matching your CUDA version), then:

```bash
pip install -r requirements-local.txt
```

The model weights (Qwen2.5-7B-Instruct, with Qwen2.5-3B-Instruct as automatic fallback if
the GPU runs out of memory) are downloaded once from Hugging Face and cached.

### Choosing the LLM provider (`.env`)

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `local` | CUDA GPU | Recommended. Runs fully on the machine, no API key, no per-request cost |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` (default `gpt-4o-mini`) |
| `mistral` | `MISTRAL_API_KEY` | `MISTRAL_MODEL` |
| `huggingface` | `HF_API_KEY` | `HF_MODEL` |
| `mock` | nothing | Demo/tests only: reuses course sentences, no real understanding |

## Running

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

Open http://localhost:8501. The interactive API documentation is at http://localhost:8000/docs.

With Docker:

```bash
cp .env.example .env
docker compose up --build
```

Data (documents, Chroma index, quizzes, results, user database) is stored in `./data`.

## API overview

| Method | Endpoint | Role |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | public |
| GET | `/auth/me` | any |
| POST | `/documents/upload` | teacher |
| GET | `/documents`, `/documents/{id}/map` | teacher |
| POST | `/quizzes/generate`, `/quizzes/{id}/publish` | teacher |
| GET | `/quizzes`, `/quizzes/{id}` | any (students: published only, no answers) |
| POST | `/quizzes/{id}/submit` | student |
| GET | `/quizzes/{id}/results` | teacher |
| GET | `/quizzes/{id}/export/pdf`, `/quizzes/{id}/export/json` | teacher |

## Security

- Passwords hashed with bcrypt; JWT (HS256) valid 8 hours. **Set a real `JWT_SECRET_KEY`**
  before any deployment: `python -c "import secrets; print(secrets.token_hex(32))"`.
- Teacher registration can require a code (`PROFESSOR_SIGNUP_CODE`).
- Role checks on every endpoint; students receive a whitelisted view of quizzes; the identity
  on a result always comes from the token.
- CSV export protected against formula injection.

## Tests and evaluation

```bash
pytest tests/ -v                                  # 150 tests, ~30 s, no GPU or API key needed
python scripts/evaluate_grading.py                # open-answer grading benchmark
python scripts/evaluate_grading.py --agent        # same, with the grading agent (uses LLM_PROVIDER)
python scripts/evaluate.py --make-template        # template for a labelled question set
```

## Known limitations

- Scanned PDFs (no OCR) and image-heavy slides give little text and poor quizzes.
- A small local model still makes mistakes (e.g. arithmetic): review questions before publishing.
- Some generated open questions have no reference answer; they are flagged in the gradebook
  and graded by the agent from the course passage, or left to the teacher.
- Manual editing of a single question is not available yet (regenerate instead).

## Author

Rayen Mannai — Data Science Engineering, ESPRIT. Summer internship at Esprit Tech,
supervised by Mrs. Amani Ayeb.
