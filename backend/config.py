"""
Configuration centrale de QuizBot.

Toutes les valeurs sont surchargeables via variables d'environnement
(ou un fichier .env chargé par python-dotenv), ce qui permet de changer
de fournisseur LLM ou de base vectorielle sans toucher au code.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv est optionnel : si absent, on lit uniquement l'environnement système.
    pass


BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    """Lit une variable d'environnement en traitant une valeur vide comme absente.

    os.getenv() ne retourne la valeur par défaut que si la variable n'existe pas
    du tout : une ligne "DATABASE_URL=" dans .env produit une chaîne vide qui
    écrase silencieusement le défaut et fait échouer le démarrage. On considère
    donc une valeur vide (ou uniquement des espaces) comme non renseignée.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


DATA_DIR = Path(_env("QUIZBOT_DATA_DIR", str(BASE_DIR / "data")))
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
QUIZZES_DIR = DATA_DIR / "quizzes"
EXPORTS_DIR = DATA_DIR / "exports"

for d in (UPLOADS_DIR, CHROMA_DIR, QUIZZES_DIR, EXPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Settings:
    # --- Fournisseur LLM : "openai", "mistral", "huggingface", "local" ou "mock" ---
    LLM_PROVIDER: str = _env("LLM_PROVIDER", "mock")
    OPENAI_API_KEY: str = _env("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    MISTRAL_API_KEY: str = _env("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = _env("MISTRAL_MODEL", "mistral-large-latest")
    HF_API_KEY: str = _env("HF_API_KEY", "")
    HF_MODEL: str = _env("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    # --- Fournisseur "local" : modèle instruct open-source exécuté entièrement
    # sur la machine (GPU), sans clé API ni appel réseau au moment de la
    # génération. Les poids sont téléchargés une seule fois depuis Hugging Face
    # Hub au premier lancement puis mis en cache localement (~/.cache/huggingface) ;
    # ce n'est pas un appel d'API distant à chaque requête, contrairement à
    # OpenAI/Mistral/HuggingFace ci-dessus.
    # Modèle par défaut : Qwen2.5-7B-Instruct (Apache 2.0, ~4-5 Go de VRAM en 4-bit
    # avec bitsandbytes). Modèle de secours : Qwen2.5-3B-Instruct (~2 Go), utilisé
    # automatiquement si le chargement du 7B échoue pour cause d'OOM.
    LOCAL_LLM_MODEL: str = _env("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LOCAL_LLM_FALLBACK_MODEL: str = _env("LOCAL_LLM_FALLBACK_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    LOCAL_LLM_MAX_NEW_TOKENS: int = int(_env("LOCAL_LLM_MAX_NEW_TOKENS", "3600"))
    # Budget de tokens réservé à CHAQUE question demandée. Le plafond effectif
    # d'un appel vaut max(LOCAL_LLM_MAX_NEW_TOKENS, num_questions * ce budget) :
    # sans cela, un quiz de 10 questions dépasse le plafond en pleine rédaction
    # et le JSON revient tronqué ("Unterminated string"), rendant tout le lot
    # inexploitable. Un QCM complet (énoncé, 4 choix, réponse, explication,
    # extrait source) pèse ~300 tokens ; 420 laisse de la marge.
    LOCAL_LLM_TOKENS_PER_QUESTION: int = int(_env("LOCAL_LLM_TOKENS_PER_QUESTION", "420"))
    # Quantification 4-bit (bitsandbytes) : réduit l'empreinte VRAM d'un modèle 7B
    # à ~4-5 Go, ce qui le rend utilisable sur un GPU de laptop grand public.
    # Mettre à "false" si vous avez assez de VRAM (>16 Go) pour du fp16 complet.
    LOCAL_LLM_LOAD_IN_4BIT: bool = _env("LOCAL_LLM_LOAD_IN_4BIT", "true").lower() == "true"

    # --- Embeddings : modèle sentence-transformers utilisé pour le RAG ---
    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = int(_env("CHUNK_SIZE_TOKENS", "800"))
    CHUNK_OVERLAP_TOKENS: int = int(_env("CHUNK_OVERLAP_TOKENS", "150"))

    # --- Retrieval ---
    TOP_K_CHUNKS: int = int(_env("TOP_K_CHUNKS", "6"))

    # --- Correction des questions ouvertes ---
    # Similarité cosinus entre la réponse et la réponse attendue. Sous le seuil
    # bas, la réponse est fausse ; au-dessus du seuil haut, juste ; entre les
    # deux, la similarité ne suffit pas à trancher : note provisoire de 0,5, à
    # confirmer par l'enseignant. Seuils calibrés sur scripts/grading_benchmark.json.
    OPEN_ANSWER_LOW: float = float(_env("OPEN_ANSWER_LOW", "0.45"))
    OPEN_ANSWER_HIGH: float = float(_env("OPEN_ANSWER_HIGH", "0.80"))

    # --- Serveur ---
    API_HOST: str = _env("API_HOST", "0.0.0.0")
    API_PORT: int = int(_env("API_PORT", "8000"))
    API_BASE_URL: str = _env("API_BASE_URL", f"http://localhost:{API_PORT}")

    # --- Répertoires (exposés aussi ici pour un accès pratique via settings.X) ---
    DATA_DIR: Path = DATA_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    CHROMA_DIR: Path = CHROMA_DIR
    QUIZZES_DIR: Path = QUIZZES_DIR
    EXPORTS_DIR: Path = EXPORTS_DIR

    # --- Authentification (JWT) ---
    # IMPORTANT : générez une vraie valeur secrète en production, ex. :
    #   python -c "import secrets; print(secrets.token_hex(32))"
    # et placez-la dans .env sous JWT_SECRET_KEY. La valeur par défaut ci-dessous
    # n'est adaptée qu'au développement local.
    JWT_SECRET_KEY: str = _env("JWT_SECRET_KEY", "dev-only-insecure-secret-change-me")
    JWT_ALGORITHM: str = _env("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
    # Code requis pour s'inscrire en tant que professeur (évite que n'importe qui
    # s'auto-attribue le rôle enseignant). Laissez vide en développement pour désactiver.
    PROFESSOR_SIGNUP_CODE: str = _env("PROFESSOR_SIGNUP_CODE", "")

    # --- Base de données (utilisateurs / authentification) ---
    # SQLite par défaut : aucune installation de serveur requise, fichier unique.
    # Pour passer à PostgreSQL en production, il suffit de changer cette URL, ex. :
    #   DATABASE_URL=postgresql://user:password@localhost:5432/quizbot
    DATABASE_URL: str = _env("DATABASE_URL", f"sqlite:///{DATA_DIR / 'quizbot.db'}")


settings = Settings()
