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
    """Génère un petit PDF de cours à usage des tests avec contenu suffisant."""
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample_course.pdf"
    c = canvas.Canvas(str(path))

    # Page 1: Introduction
    y = 750
    c.drawString(100, y, "Chapitre 1: Introduction aux reseaux de neurones")
    y -= 20
    c.drawString(100, y, "Un reseau de neurones artificiel est un modele de calcul inspire")
    y -= 20
    c.drawString(100, y, "du fonctionnement des neurones biologiques et compose de couches.")
    y -= 20
    c.drawString(100, y, "Chaque neurone recoit des entrees, calcule une somme ponderee,")
    y -= 20
    c.drawString(100, y, "puis applique une fonction d'activation non-lineaire.")

    c.showPage()

    # Page 2: Backpropagation (more substantial)
    y = 750
    c.drawString(100, y, "Chapitre 2: La retropropagation du gradient")
    y -= 20
    c.drawString(100, y, "La retropropagation ajuste les poids du reseau en propageant l'erreur.")
    y -= 20
    c.drawString(100, y, "L'algorithme calcule le gradient de la perte par rapport a chaque poids.")
    y -= 20
    c.drawString(100, y, "Ces gradients indiquent comment modifier les poids pour reduire l'erreur.")
    y -= 20
    c.drawString(100, y, "La mise a jour des poids se fait via la descente de gradient.")

    c.showPage()

    # Page 3: Optimization (more content for quality questions)
    y = 750
    c.drawString(100, y, "Chapitre 3: Optimization et regularisation")
    y -= 20
    c.drawString(100, y, "La regularisation L2 penalise les poids eleves pour eviter le surapprentissage.")
    y -= 20
    c.drawString(100, y, "Le surapprentissage survient quand un modele memorise le bruit des donnees.")
    y -= 20
    c.drawString(100, y, "L'utilisation de donnees d'entraînement et de validation aide a detecter le surapprentissage.")

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
