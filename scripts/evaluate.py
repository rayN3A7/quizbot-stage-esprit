"""
Banc d'évaluation de QuizBot.

Le pipeline a été enrichi de plusieurs mécanismes (sélection MMR, requête
dérivée du document, contrôles qualité programmatiques, agent de vérification)
dont l'effet n'avait jamais été MESURÉ — seulement constaté à l'œil. Ce script
comble ce manque : il quantifie ce que chaque mécanisme apporte réellement,
sur des données étiquetées à la main.

Trois mesures indépendantes :

  1. CONTRÔLES QUALITÉ — précision / rappel / F1 des gardes programmatiques
     face à un jugement humain (une question est-elle réellement mauvaise ?).
     C'est la mesure la plus importante : un garde trop strict jette de bonnes
     questions, un garde trop laxiste en laisse passer de fausses.

  2. RETRIEVAL — couverture du document par les passages sélectionnés, avec et
     sans MMR. Mesurée par la dissimilarité moyenne entre passages retenus
     (deux passages quasi identiques n'apportent qu'une seule idée) et par le
     nombre de pages distinctes couvertes.

  3. GÉNÉRATION — taux de questions retenues, longueur, part de QCM dont la
     réponse est ancrée dans le document.

Usage :
    python scripts/evaluate.py --dataset data/eval/questions_labeled.json
    python scripts/evaluate.py --retrieval --document-id <id>
    python scripts/evaluate.py --make-template          # crée un jeu vierge

Le jeu étiqueté est un JSON : une liste d'objets {question..., "label": "bonne"|"mauvaise",
"raison_humaine": "..."}. Voir --make-template pour un fichier de départ.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import Question, QuestionType  # noqa: E402
from backend.quiz_generator import (  # noqa: E402
    _derive_document_query,
    _mmr_select,
    _quality_reject_reason,
)


# --------------------------------------------------------------------------- #
# Utilitaires d'affichage
# --------------------------------------------------------------------------- #

def _title(text: str) -> None:
    print(f"\n{text}")
    print("=" * len(text))


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


# --------------------------------------------------------------------------- #
# 1. Évaluation des contrôles qualité
# --------------------------------------------------------------------------- #

def evaluate_quality_gates(dataset_path: Path) -> dict:
    """Compare le verdict des gardes programmatiques au jugement humain.

    Convention : la classe POSITIVE est « mauvaise question ». Un garde qui
    rejette une question que l'humain juge mauvaise est un vrai positif.

    - Précision faible => le garde jette de bonnes questions (trop strict).
    - Rappel faible    => des questions fausses passent (trop laxiste).
    """
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("Le jeu de données est vide.")

    tp = fp = tn = fn = 0
    disagreements = []

    for rec in records:
        label = rec.get("label", "").strip().lower()
        if label not in ("bonne", "mauvaise"):
            continue

        passages = [{"text": t} for t in rec.get("passages", [])]
        question = Question(
            type=QuestionType(rec.get("type", "qcm")),
            question=rec["question"],
            choices=rec.get("choices"),
            correct_choice_index=rec.get("correct_choice_index"),
            reference_answer=rec.get("reference_answer", ""),
            explanation=rec.get("explanation", ""),
            source_excerpt=rec.get("source_excerpt", ""),
        )

        reason = _quality_reject_reason(question, passages)
        predicted_bad = reason is not None
        actually_bad = label == "mauvaise"

        if predicted_bad and actually_bad:
            tp += 1
        elif predicted_bad and not actually_bad:
            fp += 1
            disagreements.append(("FAUX POSITIF (bonne question rejetée)",
                                  rec["question"][:70], reason))
        elif not predicted_bad and actually_bad:
            fn += 1
            disagreements.append(("FAUX NÉGATIF (mauvaise question acceptée)",
                                  rec["question"][:70], rec.get("raison_humaine", "")))
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    _title("1. Contrôles qualité programmatiques")
    print(f"  questions évaluées      : {total}")
    print(f"  matrice de confusion    : VP={tp}  FP={fp}  VN={tn}  FN={fn}")
    print(f"  précision (rejets justes)  : {_pct(precision)}")
    print(f"  rappel (mauvaises captées) : {_pct(recall)}")
    print(f"  F1                         : {_pct(f1)}")
    print(f"  exactitude                 : {_pct(accuracy)}")

    if disagreements:
        print(f"\n  Désaccords avec le jugement humain ({len(disagreements)}) :")
        for kind, q, why in disagreements[:10]:
            print(f"    [{kind}]")
            print(f"      question : {q}...")
            if why:
                print(f"      motif    : {why}")

    # --- Mises en garde sur l'interprétation ------------------------------- #
    # Un F1 de 100 % sur un petit jeu construit à la main ne dit presque rien :
    # il faut le signaler explicitement plutôt que de laisser lire le chiffre
    # comme une validation du pipeline.
    warnings = []
    if total < 30:
        warnings.append(
            f"jeu très petit ({total} questions) — les scores ne sont pas "
            "significatifs. Étiquetez au moins 30 à 50 questions RÉELLEMENT "
            "produites par votre pipeline pour une mesure exploitable."
        )
    if fp == 0 and fn == 0 and total < 50:
        warnings.append(
            "aucune erreur détectée : vérifiez que le jeu contient bien des cas "
            "limites (questions médiocres mais non aberrantes), sinon la mesure "
            "ne teste que les cas faciles."
        )

    # Les gardes sont syntaxiques : elles ne jugent JAMAIS la véracité d'un fait.
    # Si le jeu contient des questions étiquetées mauvaises pour un motif
    # factuel, on prévient que leur capture est fortuite.
    factual = [
        r for r in records
        if r.get("label") == "mauvaise"
        and any(w in (r.get("raison_humaine", "") or "").lower()
                for w in ("faux", "fausse", "incorrect", "contredit", "erreur factuelle"))
    ]
    if factual:
        warnings.append(
            f"{len(factual)} question(s) sont étiquetées mauvaises pour un motif "
            "FACTUEL. Les contrôles programmatiques ne jugent pas la véracité : "
            "s'ils les rejettent, c'est par un défaut de forme corrélé, pas parce "
            "qu'ils ont détecté l'erreur. Ne comptez pas ces cas comme une preuve "
            "que le pipeline détecte les faits faux."
        )

    if warnings:
        print("\n  ⚠ Précautions d'interprétation :")
        for w in warnings:
            print(f"    - {w}")

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# --------------------------------------------------------------------------- #
# 2. Évaluation du retrieval
# --------------------------------------------------------------------------- #

def _mean_pairwise_dissimilarity(passages: list[dict]) -> float:
    """Dissimilarité cosinus moyenne entre les passages retenus.

    Plus la valeur est haute, moins les passages se répètent — donc plus le
    quiz peut couvrir d'idées distinctes. C'est exactement ce que MMR cherche
    à maximiser sans sacrifier la pertinence.
    """
    from backend.embeddings import cosine_similarity

    vecs = [p["embedding"] for p in passages if p.get("embedding") is not None]
    if len(vecs) < 2:
        return float("nan")
    sims = [
        cosine_similarity(vecs[i], vecs[j])
        for i in range(len(vecs)) for j in range(i + 1, len(vecs))
    ]
    return 1.0 - (sum(sims) / len(sims))


def evaluate_retrieval(document_id: str, k: int = 6) -> dict:
    """Compare le top-k brut à la sélection MMR sur un document réel."""
    from backend.models import QuizConfig
    from backend.vectorstore import get_vector_store

    store = get_vector_store()
    config = QuizConfig(document_id=document_id, num_questions=k)
    query = _derive_document_query(document_id, config)

    candidates = store.query(document_id, query, top_k=k * 3)
    if not candidates:
        raise SystemExit(f"Aucun passage indexé pour le document {document_id}.")

    naive = candidates[:k]
    mmr = _mmr_select([dict(c) for c in candidates], k)

    def pages(ps):
        return {p.get("metadata", {}).get("source_page") for p in ps}

    naive_div = _mean_pairwise_dissimilarity(naive)
    mmr_div = _mean_pairwise_dissimilarity(mmr)

    _title("2. Retrieval : top-k brut vs MMR")
    print(f"  document                : {document_id}")
    print(f"  requête dérivée         : {query[:90]}...")
    print(f"  candidats examinés      : {len(candidates)} (pour k={k})")
    print()
    print(f"  {'':22} {'top-k brut':>12} {'MMR':>12}")
    print(f"  {'dissimilarité moyenne':22} {naive_div:>12.3f} {mmr_div:>12.3f}")
    print(f"  {'pages distinctes':22} {len(pages(naive)):>12} {len(pages(mmr)):>12}")

    if mmr_div == mmr_div and naive_div == naive_div:   # non-NaN
        gain = (mmr_div - naive_div) / naive_div if naive_div else 0.0
        print(f"\n  gain de diversité MMR   : {gain:+.1%}")

    return {"naive_diversity": naive_div, "mmr_diversity": mmr_div,
            "naive_pages": len(pages(naive)), "mmr_pages": len(pages(mmr))}


# --------------------------------------------------------------------------- #
# 3. Statistiques descriptives sur un quiz généré
# --------------------------------------------------------------------------- #

def evaluate_generated_quiz(quiz_path: Path) -> dict:
    """Mesures descriptives sur un quiz déjà généré (fichier data/quizzes/*.json)."""
    quiz = json.loads(quiz_path.read_text(encoding="utf-8"))
    questions = quiz.get("questions", [])
    if not questions:
        raise SystemExit("Ce quiz ne contient aucune question.")

    lengths = [len(q["question"]) for q in questions]
    mcq = [q for q in questions if q["type"] == "qcm"]
    with_excerpt = [q for q in questions if (q.get("source_excerpt") or "").strip()]
    with_expl = [q for q in questions if (q.get("explanation") or "").strip()]
    themes = {q.get("theme", "") for q in questions if q.get("theme")}

    _title(f"3. Quiz généré : {quiz.get('title', quiz_path.stem)}")
    print(f"  questions               : {len(questions)}  ({len(mcq)} QCM, {len(questions)-len(mcq)} ouvertes)")
    print(f"  longueur d'énoncé       : médiane {statistics.median(lengths):.0f} car."
          f"  (min {min(lengths)}, max {max(lengths)})")
    print(f"  avec extrait source     : {len(with_excerpt)}/{len(questions)}")
    print(f"  avec explication        : {len(with_expl)}/{len(questions)}")
    print(f"  thèmes distincts        : {len(themes)}")
    if quiz.get("agent_report"):
        r = quiz["agent_report"]
        print(f"  rapport de l'agent      : {r.get('generated')} générées, "
              f"{r.get('dropped')} écartées, {r.get('regenerated')} régénérées")

    return {"n": len(questions), "n_mcq": len(mcq), "themes": len(themes),
            "median_length": statistics.median(lengths)}


# --------------------------------------------------------------------------- #
# Jeu de données modèle
# --------------------------------------------------------------------------- #

TEMPLATE = [
    {
        "_commentaire": "Étiquetez chaque question à la main : label = bonne | mauvaise. "
                        "Les champs 'passages' servent aux contrôles d'ancrage.",
        "type": "qcm",
        "question": "Pourquoi utilise-t-on plusieurs arbres dans une Random Forest ?",
        "choices": [
            "Pour pallier le problème de généralisation d'un arbre unique",
            "Pour réduire de moitié le temps d'entraînement",
            "Pour supprimer le besoin de données étiquetées",
            "Pour garantir une profondeur minimale",
        ],
        "correct_choice_index": 0,
        "reference_answer": "Un arbre seul généralise mal ; l'ensemble corrige ce défaut.",
        "explanation": "Le bagging réduit la variance du modèle.",
        "source_excerpt": "Pour palier le probleme de generalisation des arbres de decision",
        "passages": ["Pour palier le probleme de generalisation des arbres de decision, "
                     "on utilise plusieurs arbres de decision : Random Forest."],
        "label": "bonne",
        "raison_humaine": "",
    },
    {
        "type": "ouverte",
        "question": "Quand a-t-on introduit la première loi de robotique ?",
        "reference_answer": "1987",
        "explanation": "Introduite en 1987 par Marvin Minsky.",
        "source_excerpt": "La loi de robotique a ete introduite en 1987 par Marvin Minsky",
        "passages": ["K-Means est un algorithme de clustering non supervise."],
        "label": "mauvaise",
        "raison_humaine": "Fait inventé, absent du cours (et faux : Asimov, 1942).",
    },
]


def make_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Jeu de données modèle écrit dans {path}")
    print("Complétez-le avec vos propres questions générées, étiquetez-les, puis relancez :")
    print(f"  python scripts/evaluate.py --dataset {path}")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Banc d'évaluation de QuizBot")
    parser.add_argument("--dataset", type=Path, help="jeu de questions étiquetées (JSON)")
    parser.add_argument("--retrieval", action="store_true", help="évaluer le retrieval (MMR)")
    parser.add_argument("--document-id", help="document à utiliser pour --retrieval")
    parser.add_argument("--quiz", type=Path, help="quiz généré à décrire (data/quizzes/*.json)")
    parser.add_argument("--k", type=int, default=6, help="nombre de passages (défaut : 6)")
    parser.add_argument("--make-template", action="store_true",
                        help="créer un jeu de données modèle à étiqueter")
    args = parser.parse_args()

    if args.make_template:
        make_template(Path("data/eval/questions_labeled.json"))
        return

    ran = False
    if args.dataset:
        evaluate_quality_gates(args.dataset)
        ran = True
    if args.retrieval:
        if not args.document_id:
            raise SystemExit("--retrieval nécessite --document-id")
        evaluate_retrieval(args.document_id, args.k)
        ran = True
    if args.quiz:
        evaluate_generated_quiz(args.quiz)
        ran = True

    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
