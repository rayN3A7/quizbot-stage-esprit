"""Tests du module d'ingestion (extraction + chunking)."""
from pathlib import Path

from backend.ingestion import (
    _merge_wrapped_lines, chunk_pages, extract_text, extract_text_from_pdf,
    normalize_typography, process_document, strip_repeated_boilerplate,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_extract_text_from_pdf(sample_pdf):
    pages = extract_text_from_pdf(sample_pdf)
    assert len(pages) == 3
    assert "reseaux de neurones" in pages[0][1]
    assert "retropropagation" in pages[1][1]
    assert "regularisation L2" in pages[2][1]


def test_chunk_pages_respects_size_and_overlap():
    pages = [(1, "mot " * 100)]
    chunks = chunk_pages(pages, chunk_size_tokens=30, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 30


def test_chunk_pages_tracks_source_page():
    pages = [(1, "a " * 10), (2, "b " * 10)]
    chunks = chunk_pages(pages, chunk_size_tokens=8, overlap_tokens=2)
    pages_seen = {c.source_page for c in chunks}
    assert pages_seen <= {1, 2}


def test_process_document_end_to_end(sample_pdf):
    chunks = process_document(sample_pdf)
    assert len(chunks) >= 1
    full_text = " ".join(c.text for c in chunks)
    assert "neurones" in full_text


# --------------------------------------------------------------------------- #
# Régression : suppression des en-têtes/pieds de page répétés
# (bug réel signalé : "ESPRIT" apparaissait comme "concept clé" généré alors
# que c'est un bandeau de copyright présent sur chaque diapositive)
# --------------------------------------------------------------------------- #

def test_strip_repeated_boilerplate_removes_copyright_lines():
    pages = [
        (1, "© 2022-2023 – ESPRIT – Module TEST\nContenu réel de la page un."),
        (2, "© 2022-2023 – ESPRIT – Module TEST\nContenu réel de la page deux."),
        (3, "© 2022-2023 – ESPRIT – Module TEST\nContenu réel de la page trois."),
    ]
    cleaned = strip_repeated_boilerplate(pages)
    for _, text in cleaned:
        assert "ESPRIT" not in text
        assert "©" not in text
    assert "Contenu réel de la page un." in cleaned[0][1]


def test_strip_repeated_boilerplate_handles_multiple_footer_variants():
    # Cas réel observé : un même document mélange deux bandeaux différents
    # selon les sections, chacun sous 50% de fréquence globale.
    pages = (
        [(i, f"© 2022-2023 – ESPRIT – Module A\nContenu section A page {i}.") for i in range(1, 6)]
        + [(i, f"© 2022-2023 – ESPRIT – Module B\nContenu section B page {i}.") for i in range(6, 9)]
    )
    cleaned = strip_repeated_boilerplate(pages)
    for _, text in cleaned:
        assert "ESPRIT" not in text
        assert "Module A" not in text
        assert "Module B" not in text


def test_strip_repeated_boilerplate_keeps_legitimate_repeated_short_doc():
    # Document trop court (< 3 pages) : on ne doit rien supprimer par excès de prudence.
    pages = [(1, "© Ligne répétée\nContenu."), (2, "© Ligne répétée\nAutre contenu.")]
    cleaned = strip_repeated_boilerplate(pages)
    # Les lignes de copyright restent supprimées (mécanisme inconditionnel),
    # mais le contenu réel doit être préservé.
    assert "Contenu." in cleaned[0][1]
    assert "Autre contenu." in cleaned[1][1]


def test_strip_repeated_boilerplate_preserves_non_repeated_content():
    pages = [
        (1, "© Copyright\nPremier sujet unique."),
        (2, "© Copyright\nDeuxième sujet totalement différent."),
        (3, "© Copyright\nTroisième sujet encore différent."),
    ]
    cleaned = strip_repeated_boilerplate(pages)
    assert "Premier sujet unique." in cleaned[0][1]
    assert "Deuxième sujet totalement différent." in cleaned[1][1]
    assert "Troisième sujet encore différent." in cleaned[2][1]


# --------------------------------------------------------------------------- #
# Régression : fusion des lignes tronquées par le retour à la ligne du PDF
# --------------------------------------------------------------------------- #

def test_merge_wrapped_lines_joins_continuation():
    text = "• artifactId : Unique name used to name the artifact to be\nbuilt."
    merged = _merge_wrapped_lines(text)
    assert merged == "• artifactId : Unique name used to name the artifact to be built."


def test_merge_wrapped_lines_keeps_separate_bullets():
    text = "• First point.\n• Second point.\n• Third point."
    merged = _merge_wrapped_lines(text)
    assert merged.count("\n") == 2
    assert "• First point." in merged
    assert "• Second point." in merged


def test_merge_wrapped_lines_starts_new_line_after_sentence_end():
    text = "First complete sentence.\nSecond complete sentence."
    merged = _merge_wrapped_lines(text)
    assert merged == "First complete sentence.\nSecond complete sentence."


# --------------------------------------------------------------------------- #
# Régression bout-en-bout sur un vrai support de cours (fixture réelle)
# --------------------------------------------------------------------------- #

def test_real_course_pdf_has_no_leaked_institutional_boilerplate():
    """
    Reproduit le bug réel signalé : sur un vrai support de cours ESPRIT (PDF
    "Spring Boot - Maven"), le nom de l'établissement et le bandeau de
    copyright ne doivent plus apparaître dans le texte indexé/chunké, quel
    que soit le nombre de variantes de bandeau utilisées dans le document.
    """
    fixture = FIXTURES_DIR / "spring_boot_maven.pdf"
    if not fixture.exists():
        import pytest
        pytest.skip("Fixture PDF réelle non présente dans cet environnement.")

    chunks = process_document(fixture)
    assert len(chunks) >= 1
    for c in chunks:
        assert "ESPRIT" not in c.text
        assert "©" not in c.text
        assert "2022-2023" not in c.text


# --------------------------------------------------------------------------- #
# Normalisation typographique — conversion Unicode → ASCII
# --------------------------------------------------------------------------- #

def test_normalize_typography_converts_unicode_minus_to_hyphen():
    """Le tiret moins mathématique (U+2212) doit être converti en tiret ASCII."""
    # Exemple réel : PDFs contiennent "2*n−1" avec U+2212 au lieu de "-"
    text = "La formule 2*n−1 est invalide en Python."
    normalized = normalize_typography(text)
    assert "2*n-1" in normalized
    assert "2*n−1" not in normalized


def test_normalize_typography_converts_all_dash_variants():
    """Tous les tirets typographiques (en/em dash, moins) → ASCII hyphen."""
    text = "Un–deux–trois — quatre — cinq−six"  # U+2013, U+2014, U+2212
    normalized = normalize_typography(text)
    assert normalized == "Un-deux-trois - quatre - cinq-six"


def test_normalize_typography_converts_smart_quotes():
    """Les guillemets typographiques doivent devenir des ASCII quotes."""
    # U+201C = " (left double quote), U+201D = " (right double quote)
    # U+2018 = ' (left single quote), U+2019 = ' (right single quote)
    text = '\u201CBonjour\u201D et \u2018salut\u2019 en typographie courbe.'
    normalized = normalize_typography(text)
    assert '"Bonjour"' in normalized
    assert "'salut'" in normalized
    assert '\u201CBonjour\u201D' not in normalized
    assert '\u2018salut\u2019' not in normalized


def test_normalize_typography_converts_special_spaces():
    """Les espaces insécables et fines doivent devenir des espaces normales."""
    text = "Mot\u00A0insécable et mot\u2009fin."  # U+00A0 et U+2009
    normalized = normalize_typography(text)
    assert normalized == "Mot insécable et mot fin."


def test_normalize_typography_converts_operator_characters():
    """L'astérisque opérateur (U+2217) et ellipsis (U+2026) → ASCII."""
    text = "2∗n et attends…"  # U+2217 et U+2026
    normalized = normalize_typography(text)
    assert "2*n" in normalized
    assert "attends..." in normalized


def test_normalize_typography_preserves_regular_ascii():
    """Les caractères ASCII normaux ne doivent pas changer."""
    text = "normal-text with 'quotes' and (parens) and ** operator."
    normalized = normalize_typography(text)
    assert normalized == text
