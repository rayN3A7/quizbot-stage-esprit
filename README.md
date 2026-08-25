# 🧠 QuizBot — Assistant Pédagogique Intelligent

Génération automatique de questionnaires pédagogiques à partir de supports de
cours (PDF / PPTX), via un pipeline **RAG (Retrieval-Augmented Generation)**
et l'API d'un grand modèle de langage (LLM).

Implémentation complète du cahier des charges technique ESPRIT — Projet de
Stage d'Été 2025-2026.

## Architecture

Architecture en 3 couches, conforme au cahier des charges :

```
┌─────────────────────┐      HTTP       ┌──────────────────────┐
│  Frontend (Streamlit) │ ─────────────▶ │  Backend (FastAPI)    │
│  - Espace Enseignant  │ ◀───────────── │  - Ingestion           │
│  - Espace Étudiant    │                │  - RAG (Chroma+LLM)   │
└─────────────────────┘                 │  - Correction         │
                                          │  - Export PDF/JSON    │
                                          └──────────────────────┘
```

**Pipeline RAG** (`backend/quiz_generator.py`) :

1. **Ingestion** (`ingestion.py`) — extraction du texte (PyMuPDF pour PDF,
   python-pptx pour PPTX). Les en-têtes/pieds de page répétés (bandeaux de
   copyright, nom d'établissement, etc.) sont automatiquement détectés et
   supprimés avant indexation, et les lignes coupées par un retour à la
   ligne du PDF sont recollées, afin que le contenu indexé — et donc les
   questions générées — ne référencent jamais du texte institutionnel ou
   des phrases tronquées plutôt que le contenu pédagogique réel. Le texte
   est ensuite découpé en chunks de ~800 tokens avec 150 tokens de
   chevauchement.
2. **Indexation** (`vectorstore.py`) — chaque chunk est vectorisé
   (`sentence-transformers`, modèle `all-MiniLM-L6-v2`) et stocké dans
   ChromaDB, une collection par document.
3. **Retrieval** — au moment de la génération, les *k* passages les plus
   pertinents sont récupérés par similarité sémantique.
4. **Génération** (`llm_client.py`) — les passages sont injectés dans un
   prompt structuré envoyé au LLM choisi (OpenAI / Mistral / Hugging Face),
   qui répond en JSON strict (questions, choix, réponses, explications).
5. **Correction** (`grading.py`) — les QCM sont corrigés par comparaison
   exacte, les questions ouvertes par similarité sémantique des embeddings
   (cosine similarity, seuil configurable).
6. **Export** (`export.py`) — génération de PDF (ReportLab) et JSON.

## Structure du projet

```
quizbot/
├── backend/
│   ├── config.py          # configuration centrale (.env)
│   ├── models.py          # schémas Pydantic (Quiz, Question, User, etc.)
│   ├── auth.py             # hachage bcrypt + JWT + contrôle d'accès par rôle
│   ├── database.py         # moteur/session SQLAlchemy (SQLite par défaut)
│   ├── db_models.py        # modèle ORM (table users)
│   ├── ingestion.py        # extraction PDF/PPTX + chunking
│   ├── embeddings.py       # wrapper sentence-transformers
│   ├── vectorstore.py      # wrapper ChromaDB
│   ├── llm_client.py       # abstraction multi-fournisseur LLM (+ mode mock)
│   ├── quiz_generator.py   # orchestration du pipeline RAG
│   ├── grading.py          # correction QCM / questions ouvertes
│   ├── export.py           # export PDF / JSON
│   ├── storage.py          # persistance (JSON pour quiz/docs, DB pour utilisateurs)
│   └── main.py             # API FastAPI
├── frontend/
│   └── app.py               # interface Streamlit (login + Enseignant / Étudiant)
├── scripts/
│   └── migrate_users_json_to_db.py  # migration ponctuelle ancien users.json -> DB
├── tests/                    # suite de tests pytest (48 tests)
├── data/                     # uploads, base Chroma, quiz, exports, quizbot.db (générés)
├── requirements.txt
├── .env.example
├── GUIDE_UTILISATION.md      # guide destiné aux enseignants (livrable)
└── README.md
```

## Installation

```bash
cd quizbot
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Éditez `.env` pour choisir votre fournisseur LLM :

- **Aucune clé API disponible / test rapide** → laissez `LLM_PROVIDER=mock`.
  Un générateur de questions factice (sans appel réseau) permet de tester
  tout le pipeline RAG (extraction → chunking → vectorisation → retrieval →
  génération → correction → export) de bout en bout, en réutilisant de
  vraies phrases extraites du document (jamais de texte inventé ou de
  bandeau institutionnel). **Limite assumée** : ce mode ne "comprend" pas le
  contenu — il ne fait que réutiliser des phrases réelles comme choix de
  QCM. Pour un usage réel (réviser un cours, générer des quiz pour des
  étudiants), utilisez un vrai fournisseur LLM ci-dessous : la qualité des
  questions et la pertinence des distracteurs seront nettement meilleures.
- **OpenAI** → `LLM_PROVIDER=openai` + `OPENAI_API_KEY=sk-...`
- **Mistral** → `LLM_PROVIDER=mistral` + `MISTRAL_API_KEY=...`
- **Hugging Face** → `LLM_PROVIDER=huggingface` + `HF_API_KEY=...`

## Lancer l'application

Terminal 1 — Backend :
```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 — Frontend :
```bash
streamlit run frontend/app.py
```

Ouvrez ensuite http://localhost:8501. La documentation interactive de l'API
est disponible sur http://localhost:8000/docs (Swagger UI généré
automatiquement par FastAPI).

## Authentification et sécurité

QuizBot distingue deux rôles : **professeur** et **étudiant**, avec une
authentification par jeton JWT.

- **Inscription** : `POST /auth/register` (username, mot de passe ≥ 6
  caractères, nom complet, rôle). L'inscription en tant que professeur peut
  être protégée par un code (`PROFESSOR_SIGNUP_CODE` dans `.env`) pour éviter
  que n'importe qui s'auto-attribue ce rôle — laissez-le vide en développement.
- **Connexion** : `POST /auth/login` (formulaire username/password) retourne
  un jeton JWT valable `ACCESS_TOKEN_EXPIRE_MINUTES` minutes (8h par défaut).
- Chaque requête protégée doit inclure l'en-tête `Authorization: Bearer <token>`.
- **Mots de passe** : hachés avec bcrypt (jamais stockés en clair).
- **Stockage** : les comptes utilisateurs sont stockés dans une vraie base de
  données relationnelle via SQLAlchemy — **SQLite par défaut** (fichier unique
  `data/quizbot.db`, aucune installation de serveur requise). Passer à
  PostgreSQL ou MySQL en production ne nécessite de changer que
  `DATABASE_URL` dans `.env` ; aucun code n'a besoin d'être modifié
  (voir `backend/database.py` et `backend/db_models.py`).
- **Contrôle d'accès par rôle** :
  - Professeur uniquement : upload de documents, génération/publication/export de quiz.
  - Étudiant uniquement : soumission des réponses à un quiz publié.
  - Le nom affiché sur un résultat de quiz provient toujours du compte
    authentifié — un étudiant ne peut pas soumettre de réponses sous une
    autre identité que la sienne.
  - Un étudiant ne peut jamais voir un quiz non publié.

⚠️ **Avant un déploiement réel**, générez une vraie clé secrète JWT :
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
et placez-la dans `.env` sous `JWT_SECRET_KEY` (ne jamais utiliser la valeur
par défaut fournie, qui n'est adaptée qu'au développement local).

## Lancer avec Docker

Une alternative à l'installation manuelle : Docker + docker-compose lancent
le backend et le frontend en une seule commande.

```bash
cp .env.example .env   # configurez LLM_PROVIDER et vos clés API si besoin
docker compose up --build
```

- Backend : http://localhost:8000/docs
- Frontend : http://localhost:8501

Les données (documents, base Chroma, quiz, exports) sont persistées dans le
dossier `./data` grâce au volume monté.

## Lancer les tests

```bash
pytest tests/ -v
```

La suite de tests (48 tests) couvre l'ingestion, le chunking, la génération
de quiz (avec le fournisseur `mock`), la correction et les exports, le
stockage utilisateurs en base de données, l'authentification (hachage,
JWT), ainsi qu'un test d'intégration bout-en-bout de l'API (upload →
génération → publication → soumission → correction → export) couvrant
également le contrôle d'accès par rôle. Les dépendances lourdes (ChromaDB,
sentence-transformers) sont simulées, et les tests utilisent un dossier de
données + une base de données temporaires (jamais ceux du développement
local).

## Flux d'utilisation

**Enseignant :**
1. Téléverse un PDF ou PPTX de cours → extraction + indexation automatiques.
2. Configure un quiz (nombre de questions, type QCM/ouvertes/mélange,
   difficulté, thèmes optionnels) → génération via le pipeline RAG.
3. Prévisualise, publie pour les étudiants, et/ou exporte en PDF/JSON.

**Étudiant :**
1. Choisit un quiz publié.
2. Répond aux questions.
3. Reçoit immédiatement son score, le détail des réponses, des explications,
   et des statistiques par thème / type de question.

## Limites connues (documentées dans le cahier des charges)

- La qualité des questions dépend de la qualité du document source ; les
  PDF scannés sans OCR ne sont pas supportés nativement.
- La correction des questions ouvertes repose sur une similarité sémantique
  qui peut ne pas capturer toutes les nuances d'une réponse étudiante.
- Les appels au LLM nécessitent une connexion Internet et engendrent des
  coûts variables selon le fournisseur (sauf en mode `mock`).

## Aller plus loin (pistes d'évolution)

- Étendre le stockage base de données (déjà en place pour les utilisateurs) aux documents/quiz/résultats, actuellement en JSON.
- Authentification enseignant/étudiant (JWT).
- Support de l'OCR pour les PDF scannés (ex. `pytesseract`).
- File d'attente asynchrone (Celery) pour les gros documents.
- Dashboard analytics pour l'enseignant (progression de la classe).
