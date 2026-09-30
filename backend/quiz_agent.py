"""
Agent de génération de quiz avec auto-vérification.

quiz_generator.generate_quiz() reste le pipeline RAG "one-shot" classique :
retrieval -> prompt -> LLM -> parsing, sans contrôle après-coup. Ce module
ajoute par-dessus une boucle agentique en trois temps :

  1. GÉNÈRE : un premier lot de questions est produit exactement comme dans
     quiz_generator.py (mêmes passages, même prompt).
  2. VÉRIFIE : chaque question est repassée au LLM, cette fois dans le rôle
     d'un "vérificateur pédagogique", à qui on donne les passages sources et
     la question + réponse indiquée. Il répond si la question est réellement
     répondable à partir des passages et si la réponse est correcte.
  3. RÉGÉNÈRE : toute question invalidée est régénérée (jusqu'à MAX_RETRIES
     fois), en indiquant explicitement au LLM la raison du rejet précédent
     pour qu'il évite le même problème. Si elle échoue toujours après les
     tentatives, elle est écartée plutôt que livrée telle quelle à l'étudiant.

Ce pipeline n'est utilisé que si l'enseignant active
QuizConfig.use_verification_agent=True (voir main.py) : il coûte
1 à 3 appels LLM supplémentaires par question, ce qui n'est pas toujours
souhaitable (coût, latence) par rapport au pipeline rapide existant.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel

from .llm_client import LLMError, generate_completion
from .models import Question, Quiz, QuizConfig
from .quiz_generator import (
    _PROMPT_MARKER_RE,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _extract_json_object,
    _parse_llm_response,
    _quality_reject_reason,
    _retrieve_passages,
)

logger = logging.getLogger(__name__)

# Nombre maximal de régénérations tentées par question rejetée avant de
# l'écarter définitivement du quiz.
MAX_RETRIES = 2

VERIFIER_SYSTEM_PROMPT = """Tu es un vérificateur pédagogique strict pour un questionnaire \
universitaire. On te donne un ou plusieurs passages de cours ainsi qu'une question générée à \
partir de ces passages, avec sa réponse attendue. Vérifie ces TROIS points :
1. La question est réellement répondable à partir des passages fournis, sans invention hors-sujet.
2. La réponse indiquée est bien correcte au vu des passages.
3. La question teste la COMPRÉHENSION d'un concept ou d'une définition — pas la mémorisation \
d'une phrase recopiée mot pour mot du cours. Si la question, les choix ou la réponse sont une \
simple citation brute du passage (pas de reformulation, aucun test de compréhension réel), \
considère-la comme INVALIDE même si elle est techniquement correcte.
Réponds uniquement par un objet JSON valide, sans texte additionnel, au format :
{"valide": true|false, "raison": "explication brève"}
"""


class VerificationResult(BaseModel):
    valide: bool
    raison: str = ""


def _passages_block(passages: List[dict]) -> str:
    return "\n\n".join(f"[PASSAGE {i + 1}]\n{p['text']}" for i, p in enumerate(passages))


def _build_verification_prompt(question: Question, passages: List[dict]) -> str:
    """Construit le prompt envoyé au LLM-vérificateur pour une question donnée.

    Le marqueur "MODE = VERIFICATION" en tête permet à MockProvider (mode dev
    sans clé API) de distinguer cet appel d'un appel de génération classique.
    """
    if question.type.value == "qcm" and question.choices:
        choices_block = "\n".join(
            f"  {chr(65 + idx)}. {choice}"
            + ("  <-- réponse indiquée comme correcte" if idx == question.correct_choice_index else "")
            for idx, choice in enumerate(question.choices)
        )
        question_block = f"Question (QCM) : {question.question}\nChoix :\n{choices_block}"
    else:
        question_block = (
            f"Question (ouverte) : {question.question}\n"
            f"Réponse attendue indiquée : {question.reference_answer}"
        )

    return f"""MODE = VERIFICATION

Passages de cours :
{_passages_block(passages)}

{question_block}

Vérifie si cette question est répondable à partir des passages ci-dessus et si la réponse \
indiquée est correcte."""


def verify_question(question: Question, passages: List[dict]) -> VerificationResult:
    """Appelle le LLM-vérificateur pour une question. Ne bloque jamais la
    génération en cas d'échec du vérificateur lui-même (panne fournisseur,
    réponse non parsable) : la question est alors conservée par défaut plutôt
    que de faire échouer tout le quiz sur un problème d'infrastructure.

    Ordre des vérifications :
    1. Portes programmatiques (déterministes, répliquables) : appelées EN PREMIER
    2. LLM vérificateur (peut être sujet à ses propres biais) : appelées après

    Raison : un petit modèle peut valider ses propres erreurs. Les vérifications
    programmatiques détectent des problèmes structurels que le LLM peut manquer.
    """
    # --- Étape 1 : portes programmatiques (avant le LLM) ---
    quality_reason = _quality_reject_reason(question, passages)
    if quality_reason is not None:
        return VerificationResult(valide=False, raison=quality_reason)

    # --- Étape 2 : LLM vérificateur ---
    prompt = _build_verification_prompt(question, passages)
    try:
        raw = generate_completion(VERIFIER_SYSTEM_PROMPT, prompt)
    except LLMError:
        return VerificationResult(
            valide=True, raison="Vérification indisponible, question conservée par défaut."
        )

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    cleaned = _extract_json_object(cleaned)
    try:
        data = json.loads(cleaned)
        return VerificationResult(valide=bool(data.get("valide", True)), raison=data.get("raison", ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return VerificationResult(
            valide=True, raison="Réponse du vérificateur non parsable, question conservée par défaut."
        )


def _build_regeneration_prompt(
    config: QuizConfig, passages: List[dict], failed_question: Question, reason: str
) -> str:
    single_question_config = QuizConfig(
        document_id=config.document_id,
        title=config.title,
        num_questions=1,
        question_type=failed_question.type,
        difficulty=failed_question.difficulty,
        themes=[failed_question.theme] if failed_question.theme else config.themes,
    )
    # Le texte rejeté peut contenir une étiquette de passage : la recopier ici la
    # ferait passer pour un vrai passage aux yeux du modèle.
    safe_reason = _PROMPT_MARKER_RE.sub("<étiquette de passage>", reason)
    safe_question = _PROMPT_MARKER_RE.sub("<étiquette de passage>", failed_question.question)
    note = (
        "\nIMPORTANT : une précédente tentative de question sur ce même contenu a été rejetée, "
        f"pour la raison suivante : {safe_reason}\n"
        f'Question rejetée : "{safe_question}"\n'
        "Génère une question DIFFÉRENTE qui évite ce problème.\n"
    )
    return _build_user_prompt(single_question_config, passages, extra_instructions=note)


def _regenerate_question(
    config: QuizConfig, passages: List[dict], failed_question: Question, reason: str
) -> Optional[Question]:
    prompt = _build_regeneration_prompt(config, passages, failed_question, reason)
    try:
        raw = generate_completion(SYSTEM_PROMPT, prompt)
        candidates = _parse_llm_response(raw)
    except LLMError:
        return None
    return candidates[0] if candidates else None


def generate_quiz_agentic(document_id: str, document_name: str, config: QuizConfig) -> Quiz:
    """Équivalent de quiz_generator.generate_quiz(), avec boucle génère ->
    vérifie -> régénère. Chaque question du lot initial passe par
    verify_question() ; en cas d'échec, jusqu'à MAX_RETRIES régénérations
    ciblées sont tentées avant d'écarter définitivement la question.

    Le rapport d'activité de l'agent (nombre régénéré / écarté / validé du
    premier coup) est stocké dans quiz.agent_report pour traçabilité côté
    enseignant (utile pour juger si le document source est trop pauvre,
    par exemple si beaucoup de questions sont écartées).
    """
    passages = _retrieve_passages(document_id, config)
    user_prompt = _build_user_prompt(config, passages)
    raw_response = generate_completion(SYSTEM_PROMPT, user_prompt)
    candidate_questions = _parse_llm_response(raw_response)

    final_questions: List[Question] = []
    report = {
        "generated": len(candidate_questions),
        "regenerated": 0,
        "dropped": 0,
        "verified_ok_first_try": 0,
    }

    drop_reasons: List[str] = []

    for question in candidate_questions:
        current = question
        accepted = False
        last_reason = ""
        for attempt in range(MAX_RETRIES + 1):
            result = verify_question(current, passages)
            if result.valide:
                if attempt == 0:
                    report["verified_ok_first_try"] += 1
                accepted = True
                break
            last_reason = result.raison
            if attempt < MAX_RETRIES:
                replacement = _regenerate_question(config, passages, current, result.raison)
                if replacement is None:
                    break
                current = replacement
                report["regenerated"] += 1
        if accepted:
            final_questions.append(current)
        else:
            report["dropped"] += 1
            drop_reasons.append(last_reason)
            logger.warning("Question écartée par l'agent : %s", last_reason)

    if not final_questions:
        examples = list(dict.fromkeys(drop_reasons))[:3]
        raise LLMError(
            f"Toutes les questions générées ({len(candidate_questions)}) ont été écartées par "
            f"l'agent de vérification, même après régénération. Exemples : {' | '.join(examples)}"
        )

    # Comme dans quiz_generator.generate_quiz() : on tronque au nombre demandé
    # si les régénérations en ont produit un peu plus que nécessaire.
    final_questions = final_questions[: config.num_questions]

    quiz = Quiz(title=config.title, document_name=document_name, questions=final_questions)
    quiz.agent_report = report
    return quiz
