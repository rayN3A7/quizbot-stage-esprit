"""Tests du module d'export (PDF / JSON)."""
import json

from backend.export import export_quiz_json, export_quiz_pdf
from backend.models import Question, QuestionType, Quiz


def _quiz():
    q1 = Question(
        type=QuestionType.MCQ, theme="Reseaux", question="Question QCM ?",
        choices=["A", "B", "C", "D"], correct_choice_index=1,
        reference_answer="B", explanation="exp",
    )
    q2 = Question(
        type=QuestionType.OPEN, theme="Gradient", question="Question ouverte ?",
        reference_answer="reponse attendue", explanation="exp2",
    )
    return Quiz(title="Quiz Test Export", document_name="cours.pdf", questions=[q1, q2])


def test_export_quiz_json_includes_answers_by_default():
    quiz = _quiz()
    path = export_quiz_json(quiz, include_answers=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "Quiz Test Export"
    assert data["questions"][0]["correct_choice_index"] == 1


def test_export_quiz_json_can_hide_answers():
    quiz = _quiz()
    path = export_quiz_json(quiz, include_answers=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "correct_choice_index" not in data["questions"][0]
    assert "reference_answer" not in data["questions"][0]


def test_export_quiz_pdf_produces_valid_pdf():
    quiz = _quiz()
    path = export_quiz_pdf(quiz)
    assert path.exists()
    assert path.stat().st_size > 0

    import fitz
    doc = fitz.open(path)
    text = doc[0].get_text()
    assert "Quiz Test Export" in text
    assert "Question QCM" in text


def test_export_quiz_pdf_renders_html_tags_as_text():
    """Cours HTML (quiz efbbc31d35) : Paragraph interprète son propre balisage,
    et <BR> faisait planter l'export (500)."""
    mcq = Question(
        type=QuestionType.MCQ, question="Que fait la balise <BR> dans <TITLE> ? <div>",
        choices=["<BR>", "<P>", "<CENTER>", "R&D <b"], correct_choice_index=0,
        reference_answer="<BR>", explanation="La balise <BR> passe à la ligne <div>",
    )
    open_q = Question(
        type=QuestionType.OPEN, question='Expliquez <INPUT type=radio name="i">',
        reference_answer='<DIV class="marge"> <H1>Titre</H1>', explanation="voir <STYLE>",
    )
    quiz = Quiz(title="Balises <HTML> & <BODY>", document_name="cours <web>.pdf",
                questions=[mcq, open_q])

    path = export_quiz_pdf(quiz)

    import fitz
    with fitz.open(path) as doc:
        text = " ".join("".join(page.get_text() for page in doc).split())
        spans = [s for page in doc for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for s in line["spans"]]
    for literal in ("Balises <HTML> & <BODY>", "cours <web>.pdf", "<BR> dans <TITLE> ? <div>",
                    "R&D <b", 'Expliquez <INPUT type=radio name="i">', '<DIV class="marge"> <H1>Titre</H1>',
                    "voir <STYLE>"):
        assert literal in text
    assert "Bold" in next(s for s in spans if "bonne réponse" in s["text"])["font"]
    assert "Oblique" in next(s for s in spans if "Réponse attendue" in s["text"])["font"]
