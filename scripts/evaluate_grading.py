"""
Banc d'évaluation de la correction des questions ouvertes.

Compare, sur scripts/grading_benchmark.json (réponses étiquetées juste /
partiel / faux), l'ancien barème — le score valait la similarité, juste à
partir de 0,62 — et la correction actuelle de backend/grading.py.

Mesures :
  - erreur moyenne en points : écart entre les points donnés et ceux de
    l'étiquette (juste 1, partiel 0,5, faux 0) ;
  - faux comptés justes / justes comptés faux ;
  - points moyens donnés aux réponses fausses, par type de réponse ;
  - notes provisoires laissées à l'enseignant.

Usage :
    python scripts/evaluate_grading.py              # ancien barème vs actuel
    python scripts/evaluate_grading.py --sweep      # balaye les seuils bas / haut
    python scripts/evaluate_grading.py --details    # une ligne par réponse

Utilise le vrai modèle d'embedding (pas de simulation) : premier lancement lent.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import grading  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.embeddings import cosine_similarity, embed_text  # noqa: E402
from backend.models import Question, QuestionType  # noqa: E402

OLD_THRESHOLD = 0.62
BENCHMARK = Path(__file__).with_name("grading_benchmark.json")


@lru_cache(maxsize=None)
def _embed(text: str):
    return embed_text(text)


def _cases():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    points = data["label_points"]
    for q in data["questions"]:
        question = Question(type=QuestionType.OPEN, question=q["question"],
                            reference_answer=q["reference_answer"], source_excerpt=q["source_excerpt"])
        for a in q["answers"]:
            similarity = max(cosine_similarity(_embed(a["text"]), _embed(q["reference_answer"])), 0.0)
            yield {"qid": q["id"], "question": question, "text": a["text"], "kind": a["kind"],
                   "label": a["label"], "expected": points[a["label"]], "similarity": similarity}


def _old(case):
    s = case["similarity"]
    return ("juste" if s >= OLD_THRESHOLD else "faux"), round(s, 3), False


def _current(case):
    with patch.object(grading, "embed_text", _embed):
        g = grading.grade_answer(case["question"], case["text"])
    verdict = "juste" if g.correct else ("partiel" if g.score == 0.5 else "faux")
    return verdict, g.score, g.needs_review


def _report(name, cases, grader):
    rows = [(c, *grader(c)) for c in cases]
    error = statistics.fmean(abs(points - c["expected"]) for c, _, points, _ in rows)
    false_right = sum(1 for c, v, _, _ in rows if c["label"] == "faux" and v == "juste")
    missed = sum(1 for c, v, _, _ in rows if c["label"] == "juste" and v == "faux")
    to_review = sum(1 for _, _, _, review in rows if review)
    exact = sum(1 for c, v, _, _ in rows if v == c["label"])
    print(f"\n== {name}")
    print(f"   verdict exact        : {exact}/{len(rows)}")
    print(f"   erreur moyenne       : {error:.3f} point par réponse")
    print(f"   faux comptés justes  : {false_right}")
    print(f"   justes comptés faux  : {missed}")
    print(f"   notes à confirmer    : {to_review}")
    by_kind = defaultdict(list)
    for c, _, points, _ in rows:
        by_kind[c["kind"]].append(points)
    print("   points moyens par type de réponse :")
    for kind, values in by_kind.items():
        print(f"      {kind:16s} {statistics.fmean(values):.2f}  (n={len(values)})")
    return rows


def _sweep(cases):
    """Seuils bas / haut : erreur moyenne et erreurs graves du barème à paliers
    (les règles — vide, énoncé recopié — s'appliquent avant les seuils)."""
    print("\n== balayage des seuils (barème à paliers, sans agent)")
    print("   bas   haut   erreur  faux->juste  juste->faux  à confirmer")
    results = []
    for low in [x / 100 for x in range(30, 61, 5)]:
        for high in [x / 100 for x in range(60, 96, 5)]:
            if high <= low:
                continue
            with patch.object(settings, "OPEN_ANSWER_LOW", low), patch.object(settings, "OPEN_ANSWER_HIGH", high):
                rows = [(c, *_current(c)) for c in cases]
            error = statistics.fmean(abs(p - c["expected"]) for c, _, p, _ in rows)
            fr = sum(1 for c, v, _, _ in rows if c["label"] == "faux" and v == "juste")
            mi = sum(1 for c, v, _, _ in rows if c["label"] == "juste" and v == "faux")
            rv = sum(1 for _, _, _, r in rows if r)
            results.append((error, fr, mi, rv, low, high))
    for error, fr, mi, rv, low, high in sorted(results)[:12]:
        print(f"   {low:.2f}  {high:.2f}   {error:.3f}   {fr:>5}        {mi:>5}        {rv:>5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    cases = list(_cases())
    print(f"{len(cases)} réponses, {len({c['qid'] for c in cases})} questions — "
          f"seuils actuels : bas {settings.OPEN_ANSWER_LOW}, haut {settings.OPEN_ANSWER_HIGH}")
    _report(f"ancien barème (points = similarité, juste dès {OLD_THRESHOLD})", cases, _old)
    rows = _report("correction actuelle", cases, _current)
    if args.details:
        print("\n   similarité  étiquette  verdict   points  réponse")
        for c, verdict, points, _ in rows:
            print(f"   {c['similarity']:.2f}        {c['label']:8s}   {verdict:8s}  {points:.1f}    {c['text'][:70]}")
    if args.sweep:
        _sweep(cases)


if __name__ == "__main__":
    main()
