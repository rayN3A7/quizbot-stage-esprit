"""Superposition des résultats étudiants sur la carte sémantique."""
from backend.semantic_map import overlay_performance

INTRO = ("Ce chapitre presente le reseau de neurones et ses couches, puis les notions "
         "de base du reseau de neurones artificiel utilisees dans la suite du cours. ")
CHUNKS = {
    0: INTRO + "Un neurone calcule une somme ponderee de ses entrees.",
    1: ("Rappels d'algebre lineaire : vecteurs, matrices et produit scalaire, utiles pour "
        "les calculs qui suivent et pour la notation employee dans ce chapitre. "
        "La retropropagation ajuste les poids du reseau en propageant l'erreur vers l'entree."),
    2: "Le surapprentissage survient quand un modele memorise le bruit des donnees.",
}
# Question reformulée (comme l'exige le prompt) : ses mots ressemblent surtout à
# l'aperçu du chunk 0, alors que l'extrait vient du milieu du chunk 1.
QUESTION = "Dans un reseau de neurones, que fait la retropropagation ?"
EXCERPT = "La retropropagation ajuste les poids du reseau en propageant l'erreur"


def _map():
    return {"points": [{"chunk_index": i, "preview": " ".join(t.split())[:160]} for i, t in CHUNKS.items()]}


def _attempts(semantic_map):
    return {p["chunk_index"]: p["attempts"] for p in semantic_map["points"]}


def _result(**graded):
    return {"graded_answers": [{"question": QUESTION, "score": 1.0, **graded}]}


def test_answer_attaches_to_the_chunk_containing_its_excerpt_beyond_the_preview():
    assert EXCERPT.lower() not in CHUNKS[1][:160].lower()
    m = overlay_performance(_map(), [_result(source_excerpt=EXCERPT)], chunk_texts=CHUNKS)
    assert _attempts(m) == {0: 0, 1: 1, 2: 0}


def test_answers_without_usable_excerpt_keep_the_previous_matching():
    old_file = _result()                              # résultat enregistré avant le champ
    label = _result(source_excerpt="[PASSAGE 1]")     # extrait inexploitable (étiquette)
    m = overlay_performance(_map(), [old_file, label], chunk_texts=CHUNKS)
    assert _attempts(m) == {0: 2, 1: 0, 2: 0}         # énoncé contre aperçu, comme avant


def test_excerpt_from_another_course_attaches_nowhere():
    other = _result(source_excerpt="La liaison covalente partage une paire d'electrons entre deux atomes")
    m = overlay_performance(_map(), [other], chunk_texts=CHUNKS)
    assert _attempts(m) == {0: 0, 1: 0, 2: 0}
    assert m["covered_points"] == 0
