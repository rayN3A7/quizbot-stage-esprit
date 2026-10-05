# QuizBot — User Guide for Teachers

## 1. What QuizBot does

QuizBot turns a course (PDF or PowerPoint) into a self-assessment quiz for your students in a
few clicks. Students take the quiz in their own space and get their score, the correction and
an explanation for every question immediately. You follow the results of the class in a
gradebook.

## 2. Create an account and log in

1. Open the application in your browser (address given by your administrator, for example
   `http://localhost:8501`).
2. Tab **Créer un compte** (Create an account): choose a username, a password (at least 6
   characters), your full name and the role **Professeur** (Teacher).
3. If your institution set a teacher registration code, enter it (ask your administrator).
4. Log in from the **Se connecter** (Log in) tab. Your session stays open for 8 hours.

Your password is stored hashed, never in clear text.

## 3. Upload a course

1. Tab **Téléverser un cours** (Upload a course).
2. Drop your file (`.pdf` or `.pptx`, 50 MB max.).
3. Wait a few seconds: the text is extracted, cleaned (repeated headers and footers are
   removed), split into segments and indexed. The document appears in the list with its
   number of segments.

> **Quality tip:** a document with real text gives much better questions. Scanned PDFs and
> slides made mostly of images or formulas are not supported yet: they produce few segments
> and poor questions.

## 4. Generate a quiz

1. Tab **Générer un quiz** (Generate a quiz) and choose the source document.
2. Set the **title**, the **number of questions** (1 to 20), the **type** (MCQ, open questions
   or mixed), the **difficulty**, and optionally the **themes** to focus on (one per line).
3. Optional: tick **Faire relire les questions par l'agent de vérification** (have the questions
   reviewed by the verification agent). Each question is re-checked against the course and
   weak ones are rewritten or removed. Generation takes longer but the quiz is more reliable.
4. Click **Générer le quiz**. With the local model, expect about a minute or more.
5. Review every question: answer, explanation and the **course excerpt** it comes from.

Questions that fail the automatic checks are removed before you see them. If too few remain,
generate again or ask for fewer questions.

## 5. Publish and export

- In **Mes quiz** (My quizzes), click **Publier** (Publish): the quiz becomes visible to
  students. An unpublished quiz is a draft that only teachers can see.
- **PDF**: ready to print or upload to your LMS. **JSON**: structured format for other tools.
  Exports from the interface include the answers and explanations; a student version
  without answers is available through the API (`include_answers=false`).

> Always read a quiz before publishing it: the model can still make mistakes, in particular
> on calculations.

## 6. Follow the results

Tab **Résultats** (Results), then choose a published quiz:

- **Class figures**: number of students, mean, median, students above 50 %, retries. Only the
  **first attempt** of each student counts; later attempts are practice.
- **Gradebook**: one row per submission. The **À vérifier** (To check) column counts the
  provisional grades you should confirm. Open a **detailed copy** to see each answer, how it was
  graded and why.
- **Download the gradebook (CSV)** for Excel (French format: `;` separator, decimal comma).
- **Question analysis**: success rate, choice distribution for MCQs and discrimination
  (does the question separate strong and weak students?). Questions to fix or to watch are
  flagged with a short explanation, for example a distractor chosen more often than the
  answer, which often means the answer key is wrong. With fewer than 5 copies these figures are
  only indicative.

### How open answers are graded

Each open answer gets **1 point (right), 0.5 (partial) or 0 (wrong)**:

- empty answers and answers that only repeat the question get 0;
- clearly close or clearly far answers are decided by comparing their meaning with the expected
  answer;
- uncertain cases are read by a grading agent, which also writes a one-sentence justification
  for the student;
- what the agent cannot settle is marked **provisional** for you to confirm, as are copies that
  try to talk to the grader.

## 7. Course map

Tab **Carte du cours** (Course map): every point is a segment of your course; close points deal
with close topics. With the performance overlay, the colour shows how students did on the
questions taken from each part. Red areas are parts of the course to explain again; grey areas
have not been evaluated yet.

## 8. FAQ

**A question is off-topic or wrong.** Regenerate the quiz, or use the *themes* field to guide
the generation. Editing a single question is not available yet.

**Does QuizBot need an Internet connection?** Not with the local model (default): everything
runs on the machine once the model has been downloaded. With OpenAI, Mistral or Hugging Face, an
Internet connection and an API key are needed.

**Who can see my published quizzes?** All students who have an account.

**Can a student see the answers before submitting?** No, not even by calling the API directly.

**Can a student submit under someone else's name?** No: the name on a result always comes from
the logged-in account.

**I forgot my password.** Self-service reset is not available yet: contact your administrator.
