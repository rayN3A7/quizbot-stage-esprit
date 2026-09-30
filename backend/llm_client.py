"""
Couche d'abstraction du LLM.

Le reste de l'application appelle uniquement `generate_completion(prompt)`.
Le fournisseur réel (OpenAI, Mistral, Hugging Face) est choisi via la
variable d'environnement LLM_PROVIDER, sans changer le code appelant.

Un fournisseur "mock" est fourni par défaut afin que l'application soit
testable immédiatement, sans clé API, pendant le développement.
"""
from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod

from .config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Retourne la réponse texte brute du modèle."""


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY manquant dans l'environnement (.env).")
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class MistralProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.MISTRAL_API_KEY:
            raise LLMError("MISTRAL_API_KEY manquant dans l'environnement (.env).")
        from mistralai import Mistral
        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.complete(
            model=settings.MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self):
        if not settings.HF_API_KEY:
            raise LLMError("HF_API_KEY manquant dans l'environnement (.env).")
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(model=settings.HF_MODEL, token=settings.HF_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nRéponds uniquement en JSON valide."
        return self._client.text_generation(full_prompt, max_new_tokens=2000, temperature=0.4)


class LocalLLMProvider(BaseLLMProvider):
    """
    Fournisseur 100% local : charge un modèle instruct open-source (7B par défaut,
    quantifié en 4-bit) directement sur le GPU de la machine via `transformers`.
    Aucune clé API, aucun appel réseau au moment de la génération — seul le
    téléchargement initial des poids (une fois, mis en cache par Hugging Face
    Hub) nécessite une connexion internet, comme pour n'importe quel modèle
    open-source qu'on installe.

    Chargé en singleton de classe (comme embeddings._get_model) : le modèle
    n'est chargé en mémoire GPU qu'une seule fois, même si plusieurs quiz sont
    générés dans la même session serveur.
    """

    _tokenizer = None
    _model = None
    _model_name = None

    def __init__(self):
        if LocalLLMProvider._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise LLMError(
                    "LLM_PROVIDER=local nécessite un GPU CUDA disponible (aucun détecté par "
                    "torch.cuda.is_available()). Vérifiez vos pilotes NVIDIA et l'installation "
                    "de torch avec support CUDA, ou utilisez un autre fournisseur."
                )

            quantization_config = None
            if settings.LOCAL_LLM_LOAD_IN_4BIT:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            LocalLLMProvider._model_name = None
            failures: list[str] = []
            for model_name in (settings.LOCAL_LLM_MODEL, settings.LOCAL_LLM_FALLBACK_MODEL):
                if not model_name:
                    continue
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, OSError) as e:
                    # VRAM insuffisante (ou modèle indisponible) : on libère ce
                    # qui a pu être alloué et on tente le modèle de repli, plutôt
                    # que de laisser l'application inutilisable.
                    failures.append(f"{model_name} ({type(e).__name__}: {e})")
                    torch.cuda.empty_cache()
                    continue
                LocalLLMProvider._tokenizer = tokenizer
                LocalLLMProvider._model = model
                LocalLLMProvider._model_name = model_name
                break

            if LocalLLMProvider._model is None:
                raise LLMError("Impossible de charger un modèle local : " + " ; ".join(failures))
            if failures:
                logger.warning(
                    "Modèle local de repli chargé : %s, à la place de %s",
                    LocalLLMProvider._model_name, " ; ".join(failures),
                )

        self._tokenizer = LocalLLMProvider._tokenizer
        self._model = LocalLLMProvider._model

    def _token_budget(self, user_prompt: str) -> int:
        """Plafond de tokens adapté à la taille du quiz demandé.

        Un plafond fixe suffit pour 3 questions mais coupe la réponse en plein
        milieu pour 10, ce qui produit un JSON tronqué et inexploitable. On lit
        donc NUM_QUESTIONS dans le prompt pour dimensionner le budget.
        """
        match = re.search(r"NUM_QUESTIONS\s*=\s*(\d+)", user_prompt)
        n = int(match.group(1)) if match else 1
        return max(
            settings.LOCAL_LLM_MAX_NEW_TOKENS,
            n * settings.LOCAL_LLM_TOKENS_PER_QUESTION,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._token_budget(user_prompt),
                temperature=0.4,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # On ne décode que les tokens générés (pas le prompt d'entrée qui précède).
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)


class MockProvider(BaseLLMProvider):
    """
    Fournisseur factice, sans appel réseau ni compréhension sémantique réelle :
    construit des questions directement à partir de vraies phrases extraites
    des passages fournis (pas de génération de texte "intelligente"). Utile
    pour développer / démontrer le pipeline RAG complet sans clé API, et pour
    les tests automatisés.

    Limite assumée : ce fournisseur ne "comprend" pas le contenu, il ne fait
    que réutiliser des phrases réelles du document. Pour des quiz de qualité
    pédagogique réelle (bons distracteurs, questions conceptuelles), il faut
    un vrai LLM (LLM_PROVIDER=openai/mistral/huggingface dans .env).
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+|\n+|•|▪|⮚|➢|»")
    _MIN_SENTENCE_LEN = 25
    _MAX_SENTENCE_LEN = 220

    def _extract_sentences(self, passage: str) -> list[str]:
        """Découpe un passage en phrases/puces réelles et filtre le bruit
        (fragments trop courts/longs, lignes sans lettres, numéros de page,
        titres de diapositive tout en majuscules)."""
        raw_parts = self._SENTENCE_SPLIT_RE.split(passage)
        sentences = []
        for part in raw_parts:
            s = part.strip(" \t-–—.,;")
            # Retire un numéro de page résiduel en tête de fragment (ex: "34 Error ...").
            s = re.sub(r"^\d{1,3}\s+(?=[A-ZÀ-Ý])", "", s)
            if not s or not any(c.isalpha() for c in s):
                continue
            if s.isdigit():
                continue
            # Filet de sécurité : jamais une ligne de copyright/bandeau
            # institutionnel, même si elle a échappé au nettoyage à l'ingestion.
            if s.startswith("©") or s.startswith("(c)"):
                continue
            if len(s) < self._MIN_SENTENCE_LEN or len(s) > self._MAX_SENTENCE_LEN:
                continue
            # Filtre les titres de diapositive (tout en majuscules, sans verbe
            # conjugué probable) : peu utiles comme "affirmation" à évaluer.
            letters = [c for c in s if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
                continue
            sentences.append(s)
        return sentences

    def _mock_verify(self, user_prompt: str) -> str:
        """
        Vérification factice utilisée par quiz_agent.py en mode LLM_PROVIDER=mock.

        MockProvider construit déjà ses questions à partir de vraies phrases
        extraites des passages (cf. generate() ci-dessous) : une vérification
        raisonnable et sans appel réseau consiste donc simplement à contrôler
        que le texte de la réponse indiquée (choix marqué correct, ou réponse
        ouverte attendue) apparaît bien tel quel dans les passages fournis
        dans CE prompt de vérification. Cela permet de tester toute la boucle
        génère -> vérifie -> régénère sans clé API, avec un résultat qui varie
        réellement selon le contenu (contrairement à un simple "toujours vrai").
        """
        passages_part, _, question_part = user_prompt.partition("Question (")
        passages_norm = re.sub(r"\s+", " ", passages_part).strip().lower()

        marked_choice = re.search(r"^\s*[A-D]\.\s*(.+?)\s*<--", question_part, re.MULTILINE)
        open_answer = re.search(r"Réponse attendue indiquée\s*:\s*(.+)", question_part)

        answer_text = marked_choice.group(1) if marked_choice else (
            open_answer.group(1) if open_answer else None
        )
        if not answer_text:
            return json.dumps(
                {"valide": True, "raison": "Vérification factice : aucune réponse à contrôler."},
                ensure_ascii=False,
            )

        # Tronqué à 60 caractères : robuste si la phrase a été légèrement
        # reformatée (retours à la ligne, espaces) entre génération et
        # vérification, tout en restant un contrôle significatif.
        answer_norm = re.sub(r"\s+", " ", answer_text).strip().lower()[:60]
        is_present = bool(answer_norm) and answer_norm in passages_norm

        return json.dumps({
            "valide": is_present,
            "raison": (
                "Réponse retrouvée telle quelle dans les passages fournis (mode mock)."
                if is_present else
                "Réponse indiquée introuvable dans les passages fournis (mode mock)."
            ),
        }, ensure_ascii=False)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "MODE = VERIFICATION" in user_prompt:
            return self._mock_verify(user_prompt)

        match = re.search(r"NUM_QUESTIONS\s*=\s*(\d+)", user_prompt)
        n = int(match.group(1)) if match else 3

        qtype_match = re.search(r"TYPE_QUESTIONS\s*=\s*(\S+)", user_prompt)
        qtype = qtype_match.group(1) if qtype_match else "qcm"

        raw_passages = re.findall(r"\[PASSAGE \d+\]\s*(.+?)(?=\[PASSAGE|\Z)", user_prompt, re.S)
        if not raw_passages:
            raw_passages = [user_prompt[:400]]

        # Sentences réelles par passage, pour piocher correct/distracteurs.
        sentences_by_passage = [self._extract_sentences(p) for p in raw_passages]
        # Toutes les phrases de tous les passages, avec leur passage d'origine
        # (pour piocher des distracteurs venant d'AUTRES passages).
        all_sentences_flat: list[tuple[int, str]] = [
            (pi, s) for pi, sents in enumerate(sentences_by_passage) for s in sents
        ]

        questions = []
        for i in range(n):
            passage_idx = i % len(raw_passages)
            passage_sentences = sentences_by_passage[passage_idx]
            fallback_text = raw_passages[passage_idx].strip().replace("\n", " ")[:200]

            correct_sentence = (
                passage_sentences[i % len(passage_sentences)] if passage_sentences else fallback_text
            )

            if qtype == "qcm":
                is_mcq = True
            elif qtype in ("ouverte", "open"):
                is_mcq = False
            else:  # "mélange" / "melange" : on alterne
                is_mcq = (i % 2 == 0)

            if is_mcq:
                # Distracteurs : vraies phrases venant d'AUTRES passages (sujets
                # différents), jamais de texte inventé ou de méta-commentaire.
                other_sentences = [s for pi, s in all_sentences_flat if pi != passage_idx and s != correct_sentence]
                random.shuffle(other_sentences)
                distractors = other_sentences[:3]

                # Filet de sécurité si le document est trop court/homogène pour
                # fournir 3 vraies phrases distinctes venant d'ailleurs.
                while len(distractors) < 3:
                    distractors.append(
                        f"Une affirmation ne figurant pas dans ce passage du cours ({len(distractors) + 1})"
                    )

                choices = [correct_sentence] + distractors
                random.shuffle(choices)
                correct_index = choices.index(correct_sentence)

                questions.append({
                    "type": "qcm",
                    "theme": "Général",
                    "difficulty": "moyen",
                    "question": f"Laquelle de ces affirmations correspond réellement au contenu du cours ? (Q{i + 1})",
                    "choices": choices,
                    "correct_choice_index": correct_index,
                    "reference_answer": correct_sentence,
                    "explanation": "Cette affirmation est tirée telle quelle du passage source ; les autres proposions proviennent d'autres sections du document.",
                    "source_excerpt": correct_sentence,
                })
            else:
                questions.append({
                    "type": "ouverte",
                    "theme": "Général",
                    "difficulty": "moyen",
                    "question": f"Expliquez avec vos propres mots l'idée suivante abordée dans le cours : « {correct_sentence} » (Q{i + 1})",
                    "reference_answer": correct_sentence,
                    "explanation": "Voir le passage source pour les éléments de réponse attendus.",
                    "source_excerpt": correct_sentence,
                })

        return json.dumps({"questions": questions}, ensure_ascii=False)


_PROVIDERS = {
    "openai": OpenAIProvider,
    "mistral": MistralProvider,
    "huggingface": HuggingFaceProvider,
    "local": LocalLLMProvider,
    "mock": MockProvider,
}


def get_llm_provider() -> BaseLLMProvider:
    provider_name = settings.LLM_PROVIDER.lower()
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise LLMError(
            f"Fournisseur LLM inconnu : '{provider_name}'. "
            f"Choix possibles : {list(_PROVIDERS)}"
        )
    return provider_cls()


def generate_completion(system_prompt: str, user_prompt: str) -> str:
    provider = get_llm_provider()
    return provider.generate(system_prompt, user_prompt)
