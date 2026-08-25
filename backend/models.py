"""Schémas de données partagés entre les modules (Pydantic)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    MCQ = "qcm"
    OPEN = "ouverte"


class Difficulty(str, Enum):
    EASY = "facile"
    MEDIUM = "moyen"
    HARD = "difficile"


class Role(str, Enum):
    PROFESSOR = "professeur"
    STUDENT = "etudiant"


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str = ""
    role: Role
    professor_code: Optional[str] = None  # requis uniquement si role == PROFESSOR


class UserPublic(BaseModel):
    id: str
    username: str
    full_name: str = ""
    role: Role
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserInDB(UserPublic):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    username: str


class TokenData(BaseModel):
    username: str
    role: Role


class Question(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    type: QuestionType
    theme: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    question: str
    # QCM uniquement
    choices: Optional[List[str]] = None
    correct_choice_index: Optional[int] = None
    # Toutes questions
    reference_answer: str
    explanation: str = ""
    source_excerpt: str = ""  # passage du cours utilisé pour générer la question (traçabilité)


class Quiz(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    title: str
    document_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    questions: List[Question] = Field(default_factory=list)
    published: bool = False
    created_by: str = ""  # username du professeur ayant généré le quiz
    # Renseigné uniquement si QuizConfig.use_verification_agent était vrai :
    # {"generated": int, "regenerated": int, "dropped": int, "verified_ok_first_try": int}
    agent_report: Optional[dict] = None


class QuizConfig(BaseModel):
    """Paramètres choisis par l'enseignant pour générer un quiz."""
    document_id: str
    title: str = "Quiz"
    num_questions: int = 5
    question_type: QuestionType | str = "mélange"  # "qcm", "ouverte" ou "mélange"
    difficulty: Difficulty = Difficulty.MEDIUM
    themes: Optional[List[str]] = None  # sous-thèmes à privilégier (optionnel)
    # Si vrai, chaque question générée passe par l'agent de vérification
    # (quiz_agent.py) avant d'être incluse dans le quiz : un second appel LLM
    # contrôle qu'elle est répondable et que la réponse indiquée est correcte,
    # avec régénération automatique en cas d'échec (voir quiz_agent.MAX_RETRIES).
    use_verification_agent: bool = False


class StudentAnswer(BaseModel):
    question_id: str
    answer: str


class SubmissionRequest(BaseModel):
    quiz_id: str
    answers: List[StudentAnswer]


class GradedAnswer(BaseModel):
    question_id: str
    question: str
    student_answer: str
    correct: bool
    score: float  # 0.0 - 1.0
    correct_answer: str
    explanation: str


class QuizResult(BaseModel):
    quiz_id: str
    student_name: str
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    graded_answers: List[GradedAnswer]
    total_score: float
    max_score: float
    percentage: float
    stats_by_theme: dict = Field(default_factory=dict)
    stats_by_type: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uploaded_by: str = ""  # username du professeur ayant téléversé le document
