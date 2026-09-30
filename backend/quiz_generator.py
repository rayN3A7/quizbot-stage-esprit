"""
Coeur du pipeline RAG : à partir de la configuration de quiz demandée par
l'enseignant, interroge la base vectorielle pour récupérer les passages
pertinents, construit un prompt structuré, appelle le LLM puis parse et
valide la réponse pour produire un objet Quiz.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from .embeddings import cosine_similarity
from .llm_client import generate_completion, LLMError
from .models import Question, Quiz, QuizConfig, QuestionType, Difficulty
from .vectorstore import get_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un assistant pédagogique expert, chargé de générer des questionnaires \
d'auto-évaluation pour des étudiants en cycle ingénieur, strictement à partir des passages de \
cours fournis.

Règles de qualité impératives :
- Chaque question doit évaluer la COMPRÉHENSION d'un concept, d'une définition, d'une relation \
de cause à effet ou d'une application concrète — jamais la simple mémorisation d'une phrase \
exacte du cours. Une bonne question reste sensée pour un étudiant qui maîtrise le concept mais \
n'a pas le passage sous les yeux.
- Ne recopie JAMAIS une phrase du passage mot pour mot, ni dans l'énoncé de la question, ni dans \
les choix, ni dans la réponse attendue : reformule toujours avec tes propres mots, comme le \
ferait un enseignant qui a compris le cours (et non un outil d'extraction de texte).
- Seule exception à la règle précédente : le champ "source_excerpt", qui sert de preuve. Il \
doit contenir une phrase recopiée MOT POUR MOT depuis un passage (au moins 6 mots consécutifs), \
jamais une reformulation, et jamais l'étiquette du passage (pas de "PASSAGE" suivi d'un numéro).
- Pour les QCM, les 3 distracteurs doivent être des confusions plausibles et conceptuellement \
proches du sujet (un autre concept de la même famille, une définition voisine mais incorrecte, \
une erreur fréquente d'étudiant) — jamais des phrases sans rapport tirées d'un autre passage, et \
jamais des extraits qui ressemblent à du texte brut copié-collé du document.
- Tu ne dois jamais inventer d'information absente des passages : reformuler n'est pas inventer, \
reste fidèle au sens du cours, seulement pas à sa formulation exacte.

Réponds uniquement avec un objet JSON valide, sans texte additionnel, au format suivant :

{
  "questions": [
    {
      "type": "qcm" | "ouverte",
      "theme": "sous-thème abordé",
      "difficulty": "facile" | "moyen" | "difficile",
      "question": "texte de la question",
      "choices": ["choix 1", "choix 2", "choix 3", "choix 4"],   // uniquement si type == "qcm"
      "correct_choice_index": 0,                                  // uniquement si type == "qcm", index 0-based
      "reference_answer": "réponse attendue complète",
      "explanation": "explication pédagogique de la réponse",
      "source_excerpt": "phrase recopiée mot pour mot depuis le passage utilisé"
    }
  ]
}
"""


def _build_user_prompt(
    config: QuizConfig, passages: List[dict], extra_instructions: str = "",
) -> str:
    # Les passages doivent rester le DERNIER bloc du prompt : tout texte placé
    # après serait lu comme faisant partie du dernier passage.
    passages_block = "\n\n".join(
        f"[PASSAGE {i+1}]\n{p['text']}" for i, p in enumerate(passages)
    )
    themes_line = f"THEMES_PRIORITAIRES = {', '.join(config.themes)}" if config.themes else ""

    difficulty_value = config.difficulty.value if hasattr(config.difficulty, "value") else config.difficulty
    question_type_value = config.question_type.value if hasattr(config.question_type, "value") else config.question_type
    language = _detect_language(passages)

    return f"""Génère un questionnaire à partir des passages de cours ci-dessous.

NUM_QUESTIONS = {config.num_questions}
TYPE_QUESTIONS = {question_type_value}
DIFFICULTE = {difficulty_value}
{themes_line}

Consignes :
- Base-toi exclusivement sur le contenu des passages fournis.
- Si TYPE_QUESTIONS == "mélange", alterne entre QCM et questions ouvertes.
- Les QCM doivent avoir exactement 4 choix, un seul correct.
- Pour chaque question, "source_excerpt" est une phrase du passage recopiée mot pour mot \
(la preuve que la question vient du cours), jamais l'étiquette du passage.
- Varie les sous-thèmes si plusieurs passages différents sont fournis.

--- EXEMPLES DE CALIBRAGE (ne recopie PAS leur contenu, seulement leur esprit) ---

Soit le passage fictif : « La normalisation min-max ramène chaque variable dans
l'intervalle [0,1] en soustrayant le minimum puis en divisant par l'étendue.
Elle est sensible aux valeurs aberrantes. »

MAUVAISE question (simple recopie, teste la mémoire d'une phrase) :
{{"question": "Que dit le cours sur la normalisation min-max ?",
  "choices": ["La normalisation min-max ramène chaque variable dans l'intervalle [0,1] en soustrayant le minimum puis en divisant par l'étendue", "Le tri rapide a une complexité moyenne en n log n", "Un pointeur stocke une adresse mémoire", "Le protocole HTTP est sans état"],
  "correct_choice_index": 0}}
Pourquoi elle est mauvaise : la bonne réponse est la phrase du cours copiée
telle quelle, et les distracteurs parlent d'autres sujets — on reconnaît la
réponse sans rien comprendre.

BONNE question (teste la compréhension, distracteurs plausibles) :
{{"question": "Pourquoi la normalisation min-max est-elle déconseillée sur un jeu de données contenant des valeurs extrêmes ?",
  "choices": ["Parce qu'une valeur aberrante étire l'étendue et comprime toutes les autres valeurs près de 0", "Parce qu'elle ne fonctionne que sur des variables catégorielles", "Parce qu'elle exige une distribution normale des données", "Parce qu'elle supprime automatiquement les valeurs extrêmes"],
  "correct_choice_index": 0,
  "reference_answer": "L'étendue étant calculée à partir du minimum et du maximum, une valeur aberrante l'élargit fortement et tasse les valeurs normales dans une plage étroite.",
  "source_excerpt": "Elle est sensible aux valeurs aberrantes."}}
Pourquoi elle est bonne : l'énoncé est reformulé, il demande un raisonnement
(la conséquence d'une propriété), et les trois distracteurs sont des confusions
crédibles d'étudiant sur ce même concept. Le source_excerpt, lui, est recopié
exactement depuis le passage : c'est la preuve, pas une reformulation.

--- FIN DES EXEMPLES ---

Rappel final : LANGUE = {language}. Rédige TOUTES les questions, tous les choix,
toutes les réponses et toutes les explications dans cette langue, quelle que
soit la langue des exemples ci-dessus. Seul source_excerpt reste dans la langue
du passage, recopié tel quel.
{extra_instructions}
Passages de cours :
{passages_block}
"""


def _detect_language(passages: List[dict]) -> str:
    """Devine la langue dominante des passages (français ou anglais).

    Les supports de cours en cycle ingénieur sont souvent bilingues, et un
    modèle de petite taille dérive facilement vers l'anglais même prompté en
    français. On lui indique donc explicitement la langue à employer, déduite
    du document plutôt que supposée.
    """
    sample = " ".join(p["text"] for p in passages[:4]).lower()
    fr_markers = (" le ", " la ", " les ", " des ", " est ", " une ", " dans ", " pour ", " qui ")
    en_markers = (" the ", " and ", " with ", " that ", " for ", " this ", " are ", " which ")
    fr = sum(sample.count(m) for m in fr_markers)
    en = sum(sample.count(m) for m in en_markers)
    return "anglais" if en > fr else "français"


_STOPWORDS = {
    # français
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "que", "qui", "quoi",
    "dans", "pour", "par", "sur", "avec", "sans", "sous", "est", "sont", "être", "avoir",
    "ce", "cet", "cette", "ces", "il", "elle", "ils", "elles", "on", "nous", "vous",
    "plus", "moins", "aussi", "donc", "mais", "car", "si", "au", "aux", "en", "son", "sa",
    "ses", "leur", "leurs", "peut", "doit", "fait", "faire", "tout", "tous", "toute",
    "chaque", "entre", "après", "avant", "comme", "exemple", "cours", "chapitre", "page",
    # anglais (supports bilingues fréquents en cycle ingénieur)
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "can",
    "will", "has", "have", "not", "but", "you", "your", "its", "their", "which", "when",
    "what", "how", "all", "any", "each", "example", "chapter", "page", "slide",
}

_TERM_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_\-]{2,}")


def _derive_document_query(document_id: str, config: QuizConfig) -> str:
    """Construit une requête de retrieval CARACTÉRISTIQUE DU DOCUMENT.

    Auparavant, en l'absence de thèmes fournis par l'enseignant, tous les quiz
    de tous les documents utilisaient la même requête générique ("Concepts clés
    et définitions importantes du cours"). On ne retrouvait donc pas les
    concepts du document, mais les passages qui ressemblent le plus à cette
    phrase — au hasard du contenu.

    On extrait ici les termes les plus saillants du document lui-même, par une
    heuristique TF simple pondérée par la dispersion : un terme qui revient
    souvent ET dans plusieurs chunks distincts caractérise le cours, alors
    qu'un terme concentré sur un seul chunk est probablement anecdotique.

    Retombe sur la requête générique si le document est trop court ou si
    l'extraction ne donne rien d'exploitable.
    """
    generic = "Concepts clés et définitions importantes du cours"
    if config.themes:
        return f"Concepts clés concernant : {', '.join(config.themes)}"

    store = get_vector_store()
    texts = store.all_texts(document_id)
    if len(texts) < 2:
        return generic

    freq: dict[str, int] = {}
    spread: dict[str, set] = {}
    for idx, text in enumerate(texts):
        for raw in _TERM_RE.findall(text):
            term = raw.lower()
            if term in _STOPWORDS or len(term) < 4:
                continue
            freq[term] = freq.get(term, 0) + 1
            spread.setdefault(term, set()).add(idx)

    if not freq:
        return generic

    # Score = fréquence x nombre de chunks distincts où le terme apparaît.
    # Un terme doit apparaître dans au moins 2 chunks pour compter comme thème.
    scored = [
        (f * len(spread[t]), t) for t, f in freq.items() if len(spread[t]) >= 2
    ]
    if not scored:
        return generic

    scored.sort(reverse=True)
    keywords = [t for _, t in scored[:10]]
    return "Concepts, définitions et notions clés portant sur : " + ", ".join(keywords)


def _mmr_select(hits: List[dict], top_k: int, lambda_mult: float = 0.55) -> List[dict]:
    """Sélection MMR (Maximal Marginal Relevance) parmi des passages candidats.

    Le top-k brut d'une base vectorielle renvoie les passages les PLUS PROCHES
    de la requête — or, avec un chevauchement de 150 tokens entre chunks
    voisins, ces passages sont souvent quasi identiques entre eux. Le quiz se
    concentre alors sur une seule section du cours.

    MMR choisit chaque passage suivant pour qu'il soit à la fois pertinent
    (proche de la requête) et NOUVEAU (éloigné de ceux déjà retenus) :

        score = λ · pertinence − (1 − λ) · redondance_max

    λ a été calibré sur un cas simulant 4 sections de cours de 3 chunks
    chevauchants chacune : à λ=0.55 les 4 sections sont couvertes, à λ=0.65
    seulement 3, et à λ≥0.85 le résultat redevient identique au top-k brut
    (2 sections). 0.55 garde donc la pertinence en tête de liste tout en
    couvrant l'ensemble du document.
    Sans embeddings disponibles (ex. ancienne collection Chroma), on retombe
    simplement sur l'ordre d'origine.
    """
    if not hits or top_k >= len(hits):
        return hits[:top_k]
    if any(h.get("embedding") is None for h in hits):
        return hits[:top_k]

    # distance cosinus -> pertinence (Chroma renvoie une distance, pas un score)
    for h in hits:
        h["_relevance"] = 1.0 - float(h.get("distance") or 0.0)

    selected: List[dict] = [max(hits, key=lambda h: h["_relevance"])]
    remaining = [h for h in hits if h is not selected[0]]

    while remaining and len(selected) < top_k:
        best, best_score = None, float("-inf")
        for cand in remaining:
            redundancy = max(
                cosine_similarity(cand["embedding"], s["embedding"]) for s in selected
            )
            score = lambda_mult * cand["_relevance"] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = cand, score
        selected.append(best)
        remaining.remove(best)

    for h in selected:
        h.pop("_relevance", None)
    return selected


def _retrieve_passages(document_id: str, config: QuizConfig) -> List[dict]:
    store = get_vector_store()
    query = _derive_document_query(document_id, config)

    # On récupère largement plus de candidats que nécessaire : MMR a besoin de
    # marge pour écarter les quasi-doublons sans manquer de passages.
    top_k = max(config.num_questions, 4)
    candidates = store.query(document_id, query, top_k=top_k * 3)
    if not candidates:
        raise ValueError(
            "Aucun passage indexé pour ce document. Vérifiez qu'il a bien été téléversé et traité."
        )
    return _mmr_select(candidates, top_k)


def _extract_json_object(text: str) -> str:
    """Extrait le premier objet JSON top-level d'une réponse pouvant contenir du
    texte parasite autour (fréquent avec de petits modèles locaux qui ne
    respectent pas toujours parfaitement la consigne "JSON uniquement", contrairement
    aux gros modèles cloud). Recherche la première '{' puis compte les accolades
    pour trouver sa '}' correspondante. Retourne le texte d'origine si aucune paire
    n'est trouvée (le json.loads suivant produira alors une erreur explicite)."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text


def _salvage_question_objects(text: str) -> list[dict]:
    """Récupère les objets question COMPLETS d'une réponse JSON tronquée.

    Quand le modèle atteint son plafond de tokens en pleine rédaction, la
    réponse s'arrête au milieu d'une chaîne : json.loads échoue sur l'ensemble,
    alors que les premières questions, elles, sont parfaitement formées. Plutôt
    que de perdre tout le lot, on parcourt le tableau "questions" en comptant
    les accolades (en ignorant celles situées dans une chaîne, et les
    échappements) pour extraire un par un les objets refermés.

    Retourne une liste éventuellement vide ; l'appelant décide quoi en faire.
    """
    anchor = text.find('"questions"')
    if anchor == -1:
        return []
    start = text.find("[", anchor)
    if start == -1:
        return []

    objects: list[dict] = []
    depth = 0
    obj_start = -1
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                try:
                    objects.append(json.loads(text[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = -1
        elif ch == "]" and depth == 0:
            break

    return objects


def _normalize_for_comparison(text: str) -> str:
    """Normalise le texte pour la comparaison : minuscules, ponctuation → espaces."""
    text = text.lower()
    # Remplace la plupart des caractères de ponctuation par des espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_code_like(text: str) -> bool:
    """Détecte si un texte ressemble à du code (contient tokens Python/programming)."""
    code_tokens = ['(', '[', 'lambda', 'range', '==', '!=', '**', '"def ', 'def ']
    return any(token in text for token in code_tokens)


# Tolère les fautes de frappe observées sur le modèle local : [PASSEAGE 1], [PASSEAU 2].
_PROMPT_MARKER_RE = re.compile(
    r"\[PASSAGES?\s*\d+\]|\[PASSEAU.*?\]|\[PASSEAGE.*?\]|NUM_QUESTIONS|TYPE_QUESTIONS",
    re.IGNORECASE,
)


def _quote(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return f'"{text[:limit]}…"' if len(text) > limit else f'"{text}"'


def _quality_reject_reason(question: Question, passages: List[dict]) -> str | None:
    """Vérifie une question et retourne la raison du rejet, ou None si valide.

    Ordre des vérifications : structure d'abord (fast fail), puis contenu.
    """
    # --- d) STRUCTURE (fast checks first) ---
    if question.type == QuestionType.MCQ:
        if not question.choices or len(question.choices) < 3:
            return "QCM doit avoir au moins 3 choix"
        if question.correct_choice_index is None or \
           question.correct_choice_index < 0 or \
           question.correct_choice_index >= len(question.choices):
            return "Index de la réponse correcte hors limites"
        if any(c is None or c.strip() == "" for c in question.choices):
            return "Choix vide détecté"

    # --- b) PROMPT ARTIFACTS ---
    fields = [
        ("question", question.question),
        ("reference_answer", question.reference_answer),
        ("source_excerpt", question.source_excerpt or ""),
    ] + [(f"choices[{i}]", c) for i, c in enumerate(question.choices or [])]
    for name, value in fields:
        if _PROMPT_MARKER_RE.search(value):
            return f"Marqueur de prompt dans {name} : {_quote(value)}"

    # --- a) GROUNDING (source_excerpt doit être dans les passages) ---
    if question.source_excerpt and len(question.source_excerpt.split()) >= 4:
        excerpt_norm = _normalize_for_comparison(question.source_excerpt)
        # Combine tous les passages normalisés
        passages_text = " ".join(p.get("text", "") for p in passages)
        passages_norm = _normalize_for_comparison(passages_text)

        # Cherche une fenêtre de 6 mots consécutifs
        excerpt_words = excerpt_norm.split()
        window_size = min(6, len(excerpt_words))
        found = False
        for i in range(len(excerpt_words) - window_size + 1):
            window = " ".join(excerpt_words[i:i+window_size])
            if window in passages_norm:
                found = True
                break
        if not found and window_size >= 1:
            return (
                "source_excerpt introuvable dans les passages (grounding failure) : "
                f"{_quote(question.source_excerpt)}"
            )

    # --- c) DEGENERATE CHOICES (pour MCQ) ---
    if question.type == QuestionType.MCQ and question.choices:
        choices_str = question.choices

        # BRANCH 1 : Détecte si c'est du code AVANT toute normalization
        is_code = sum(1 for c in choices_str if _is_code_like(c)) >= len(choices_str) // 2

        if is_code:
            # Pour du code : compare RAW strings uniquement, rejette exact duplicates
            if len(set(choices_str)) < len(choices_str):
                return "Choix en code : doublons exacts détectés"
        else:
            # BRANCH 2 : Non-code, peut normaliser
            normalized_choices = [_normalize_for_comparison(c) for c in choices_str]

            # Rejette exact duplicates après normalisation
            if len(set(normalized_choices)) < len(normalized_choices):
                return "Choix : doublons détectés (après normalisation)"

            # Rejette permutations (même ensemble de mots longs)
            choice_word_sets = [set(c.split()) for c in normalized_choices]
            for i, set_i in enumerate(choice_word_sets):
                for j in range(i+1, len(choice_word_sets)):
                    set_j = choice_word_sets[j]
                    # Même ensemble de mots (longueur >3) = permutation
                    long_words_i = {w for w in set_i if len(w) > 3}
                    long_words_j = {w for w in set_j if len(w) > 3}
                    if long_words_i and long_words_i == long_words_j:
                        return "Choix : permutations de mêmes concepts détectées"

            # Rejette si chevauchement moyen > 0.6
            def word_overlap(set1: set, set2: set) -> float:
                if not set1 and not set2:
                    return 0.0
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                return intersection / union if union > 0 else 0.0

            overlaps = []
            for i in range(len(choice_word_sets)):
                for j in range(i+1, len(choice_word_sets)):
                    overlaps.append(word_overlap(choice_word_sets[i], choice_word_sets[j]))

            if overlaps and sum(overlaps) / len(overlaps) > 0.6:
                return "Choix : chevauchement lexical trop élevé (>0.6 en moyenne)"

    # --- e) INTERNAL CONTRADICTION ---
    if question.type == QuestionType.MCQ and question.choices and \
       question.correct_choice_index is not None:
        answer_text = question.choices[question.correct_choice_index]
        if "**" in answer_text:
            # La réponse marquée contient **, vérifie que question ou explication aussi
            question_has_op = "**" in question.question
            explanation_has_op = "**" in question.explanation
            if not question_has_op and not explanation_has_op:
                return "Contradiction interne : réponse contient **, pas la question/explication"

    return None


def _parse_llm_response(raw: str) -> List[Question]:
    # Le modèle peut parfois entourer le JSON de ```json ... ``` malgré la consigne : on nettoie.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    cleaned = _extract_json_object(cleaned)
    try:
        data = json.loads(cleaned)
        raw_questions = data.get("questions", [])
    except json.JSONDecodeError:
        # Réponse tronquée (plafond de tokens atteint) ou légèrement malformée :
        # on récupère les questions complètes plutôt que de tout jeter.
        raw_questions = _salvage_question_objects(cleaned)
        if not raw_questions:
            raise LLMError(
                "La réponse du modèle n'est pas un JSON exploitable et aucune question "
                "complète n'a pu en être extraite. Réessayez, éventuellement avec moins "
                f"de questions. Début de la réponse : {raw[:300]}"
            )

    if not raw_questions:
        raise LLMError("Le LLM n'a retourné aucune question.")

    questions: List[Question] = []
    for q in raw_questions:
        try:
            q_type = QuestionType(q["type"])
            difficulty = Difficulty(q.get("difficulty", "moyen"))
            questions.append(Question(
                type=q_type,
                theme=q.get("theme", ""),
                difficulty=difficulty,
                question=q["question"],
                choices=q.get("choices") if q_type == QuestionType.MCQ else None,
                correct_choice_index=q.get("correct_choice_index") if q_type == QuestionType.MCQ else None,
                reference_answer=q.get("reference_answer", ""),
                explanation=q.get("explanation", ""),
                source_excerpt=q.get("source_excerpt", ""),
            ))
        except (KeyError, ValueError) as e:
            # On ignore une question malformée plutôt que de faire échouer tout le quiz.
            continue

    if not questions:
        raise LLMError("Aucune question valide n'a pu être extraite de la réponse du LLM.")
    return questions


def filter_questions_by_quality(questions: List[Question], passages: List[dict]) -> tuple[List[Question], dict]:
    """Filtre les questions par les critères de qualité et retourne (gardées, raisons_rejet).

    Returns:
        (kept_questions, rejection_reasons_dict) où rejection_reasons_dict[i] = reason
        pour chaque question rejetée (indexée par position originale).
    """
    kept = []
    reasons = {}
    for i, q in enumerate(questions):
        reason = _quality_reject_reason(q, passages)
        if reason is None:
            kept.append(q)
        else:
            reasons[i] = reason
    return kept, reasons


def generate_quiz(document_id: str, document_name: str, config: QuizConfig) -> Quiz:
    """Point d'entrée principal du pipeline RAG : retrieval -> prompt -> LLM -> parsing.

    Applique les portes de qualité programmatiques pour valider chaque question.
    """
    passages = _retrieve_passages(document_id, config)
    user_prompt = _build_user_prompt(config, passages)
    raw_response = generate_completion(SYSTEM_PROMPT, user_prompt)
    questions = _parse_llm_response(raw_response)

    kept_questions, rejection_reasons = filter_questions_by_quality(questions, passages)
    for idx, reason in rejection_reasons.items():
        logger.warning("Question %d/%d rejetée : %s", idx + 1, len(questions), reason)

    if not kept_questions:
        logger.warning("Aucune question retenue. Début de la réponse brute du modèle : %s",
                       raw_response[:500])
        examples = list(dict.fromkeys(rejection_reasons.values()))[:3]
        raise LLMError(
            f"Toutes les questions générées ({len(questions)}) ont été rejetées par les portes "
            f"de qualité. Exemples : {' | '.join(examples)}"
        )

    # On tronque/complète pour respecter au mieux le nombre de questions demandé.
    final_questions = kept_questions[: config.num_questions] if len(kept_questions) > config.num_questions else kept_questions

    # Popule le rapport d'activité (utilisé principalement par quiz_agent, mais présent aussi ici)
    quiz = Quiz(title=config.title, document_name=document_name, questions=final_questions)
    quiz.agent_report = {
        "generated": len(questions),
        "verified_ok_first_try": len(kept_questions),
        "regenerated": 0,
        "dropped": len(rejection_reasons),
    }
    return quiz
