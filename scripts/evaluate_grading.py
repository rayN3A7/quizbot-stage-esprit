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
    python scripts/evaluate_grading.py              # ancien barème vs actuel, sans agent
    python scripts/evaluate_grading.py --sweep      # balaye les seuils bas / haut
    python scripts/evaluate_grading.py --details    # une ligne par réponse
    python scripts/evaluate_grading.py --agent      # + agent de correction (LLM_PROVIDER)

Avec --agent, le modèle configuré (LLM_PROVIDER) juge chaque réponse une fois ;
ses verdicts sont gardés, puis la vraie fonction grade_answer est rejouée avec
eux pour chaque paire de seuils : le balayage tient compte de l'agent sans le
rappeler. Le modèle d'embedding est le vrai (pas de simulation) : premier
lancement lent.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import grading  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.embeddings import cosine_similarity, embed_text  # noqa: E402
from backend.grading_agent import OpenAnswerJudge  # noqa: E402
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


def _current(case, judge=None):
    with patch.object(grading, "embed_text", _embed):
        g = grading.grade_answer(case["question"], case["text"], judge)
    verdict = "juste" if g.correct else ("partiel" if g.score == 0.5 else "faux")
    return verdict, g.score, g.needs_review


def _agent_verdicts(cases):
    """Un appel au modèle par réponse ; renvoie un juge qui rejoue ces verdicts."""
    judge = OpenAnswerJudge(max_calls=10 ** 6)
    verdicts, timings = {}, []
    for c in cases:
        started = time.perf_counter()
        verdicts[(c["question"].question, c["text"].strip())] = judge(c["question"], c["text"].strip())
        timings.append(time.perf_counter() - started)
    decided = [(c, verdicts[(c["question"].question, c["text"].strip())]) for c in cases]
    usable = [(c, d) for c, d in decided if d is not None]
    print(f"\n== agent seul ({settings.LLM_PROVIDER}), sur les {len(cases)} réponses")
    print(f"   verdicts exploitables : {len(usable)}/{len(cases)}")
    print(f"   verdict exact         : {sum(1 for c, d in usable if d[0] == c['label'])}/{len(usable)}")
    confusion = Counter((c["label"], d[0]) for c, d in usable)
    print("   étiquette -> verdict : " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(confusion.items())))
    print(f"   durée par appel       : médiane {statistics.median(timings):.1f} s, max {max(timings):.1f} s")

    def replay(question, answer):
        return verdicts.get((question.question, answer.strip()))
    return replay


def _report(name, cases, grader, judge=None):
    rows = [(c, *(grader(c, judge) if judge else grader(c))) for c in cases]
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


def _sweep(cases, judge=None):
    """Seuils bas / haut : erreur moyenne et erreurs graves du barème à paliers
    (les règles — vide, énoncé recopié — s'appliquent avant les seuils)."""
    print(f"\n== balayage des seuils ({'avec' if judge else 'sans'} agent)")
    print("   bas   haut   erreur  faux->juste  juste->faux  à confirmer")
    results = []
    for low in [x / 100 for x in range(30, 61, 5)]:
        for high in [x / 100 for x in range(60, 96, 5)]:
            if high <= low:
                continue
            with patch.object(settings, "OPEN_ANSWER_LOW", low), patch.object(settings, "OPEN_ANSWER_HIGH", high):
                rows = [(c, *_current(c, judge)) for c in cases]
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
    parser.add_argument("--agent", action="store_true")
    args = parser.parse_args()

    cases = list(_cases())
    print(f"{len(cases)} réponses, {len({c['qid'] for c in cases})} questions — "
          f"seuils actuels : bas {settings.OPEN_ANSWER_LOW}, haut {settings.OPEN_ANSWER_HIGH}")
    _report(f"ancien barème (points = similarité, juste dès {OLD_THRESHOLD})", cases, _old)
    rows = _report("correction actuelle, sans agent", cases, _current)
    judge = _agent_verdicts(cases) if args.agent else None
    if judge:
        rows = _report("correction actuelle, avec agent", cases, _current, judge)
    if args.details:
        print("\n   similarité  étiquette  verdict   points  réponse")
        for c, verdict, points, _ in rows:
            print(f"   {c['similarity']:.2f}        {c['label']:8s}   {verdict:8s}  {points:.1f}    {c['text'][:70]}")
    if args.sweep:
        _sweep(cases, judge)


if __name__ == "__main__":
    main()
