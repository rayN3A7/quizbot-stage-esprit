"""Fixtures partagées pour les tests QuizBot."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# IMPORTANT : ces variables d'environnement doivent être fixées AVANT le tout
# premier `import backend...` de la session de tests (conftest.py est chargé
# en premier par pytest), afin que les tests utilisent un dossier de données
# et une base de données temporaires, jamais ceux du développement local.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="quizbot_test_data_")
os.environ["QUIZBOT_DATA_DIR"] = _TEST_DATA_DIR
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEST_DATA_DIR) / 'test_quizbot.db'}"

import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Génère un petit PDF de cours à usage des tests."""
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample_course.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, "Chapitre 1: Introduction aux reseaux de neurones")
    c.drawString(100, 730, "Un reseau de neurones artificiel est un modele de calcul inspire")
    c.drawString(100, 710, "du fonctionnement des neurones biologiques et compose de couches.")
    c.showPage()
    c.drawString(100, 750, "Chapitre 2: La retropropagation du gradient")
    c.drawString(100, 730, "La retropropagation ajuste les poids du reseau en propageant l'erreur.")
    c.save()
    return path


@pytest.fixture
def fake_embed_text():
    """Fonction d'embedding factice (bag-of-words) pour tester sans télécharger de modèle."""
    import numpy as np

    def _fake(text: str):
        vec = np.zeros(64)
        for w in text.lower().split():
            vec[hash(w) % 64] += 1
        return vec.tolist()

    return _fake
