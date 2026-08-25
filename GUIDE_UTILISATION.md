# Guide d'utilisation — QuizBot

*Guide synthétique à destination des enseignants*

## 1. À quoi sert QuizBot ?

QuizBot vous permet de transformer automatiquement un support de cours (PDF
ou diapositives PowerPoint) en questionnaire d'auto-évaluation pour vos
étudiants, en quelques clics — sans avoir à rédiger vous-même les questions.

## 2. Créer votre compte et vous connecter

1. Ouvrez l'application dans votre navigateur (adresse fournie par votre
   administrateur, ex. `http://localhost:8501`).
2. Onglet **📝 Inscription** : choisissez un nom d'utilisateur, un mot de
   passe (6 caractères minimum), votre nom complet, et le rôle **👩‍🏫
   Professeur**.
3. Si votre établissement a activé un code d'inscription professeur,
   saisissez-le (demandez-le à votre administrateur système).
4. Une fois le compte créé, allez dans l'onglet **🔑 Connexion** pour vous
   authentifier.

> 🔒 Votre mot de passe est haché (jamais stocké en clair) et votre session
> reste active pendant plusieurs heures avant de devoir vous reconnecter.

## 3. Téléverser un support de cours

1. Onglet **1. Téléverser un cours**.
2. Cliquez sur la zone de dépôt et sélectionnez votre fichier (`.pdf` ou
   `.pptx`).
3. Cliquez sur **Téléverser et indexer**.
4. Patientez quelques secondes : le système extrait le texte, le découpe
   en segments et les indexe pour la recherche sémantique.

> 💡 **Conseil qualité** : un PDF bien structuré (texte réel, pas une image
> scannée) donne de meilleurs résultats. Les documents scannés sans OCR ou
> contenant surtout des équations mathématiques complexes ne sont pas
> encore pris en charge.

## 4. Générer un quiz

1. Onglet **2. Générer un quiz**.
2. Choisissez le document source dans la liste déroulante.
3. Configurez :
   - **Titre du quiz**
   - **Nombre de questions** (1 à 20)
   - **Type de questions** : QCM, questions ouvertes, ou mélange
   - **Niveau de difficulté** : facile, moyen, difficile
   - **Thèmes spécifiques** (optionnel) : indiquez un ou plusieurs
     sous-thèmes à privilégier, un par ligne
4. Cliquez sur **🚀 Générer le quiz**.
5. Prévisualisez chaque question générée (question, choix, bonne réponse,
   explication, extrait source du cours utilisé pour la traçabilité).

## 5. Publier le quiz pour les étudiants

Une fois satisfait du contenu, cliquez sur **📢 Publier ce quiz pour les
étudiants**. Le quiz devient immédiatement visible dans l'espace étudiant.

> Un quiz non publié reste un brouillon : vous seul le voyez, et vous
> pouvez le régénérer si le résultat ne vous convient pas.

## 6. Exporter un quiz

Depuis l'onglet **3. Gérer mes quiz**, chaque quiz propose deux boutons :

- **📄 Export PDF** : document prêt à imprimer ou à déposer sur votre ENT.
- **🗂️ Export JSON** : format structuré pour intégration avec d'autres
  outils (LMS, tableur, etc.).

## 7. Suivre les résultats des étudiants

Chaque étudiant reçoit son score et le détail de ses réponses instantanément
après avoir soumis le quiz. Les résultats sont également enregistrés côté
serveur (dossier `data/results/`) pour consultation ultérieure.

## 8. Questions fréquentes

**Le quiz généré contient une question hors sujet, que faire ?**
Régénérez le quiz (les résultats varient d'une génération à l'autre) ou
affinez le champ *Thèmes spécifiques* pour orienter la génération.

**Puis-je modifier une question après génération ?**
La version actuelle permet la prévisualisation et la régénération ; l'édition
manuelle d'une question individuelle est une évolution prévue (voir README,
section "Aller plus loin").

**Le système fonctionne-t-il sans connexion Internet ?**
Non, sauf en mode de démonstration interne (`LLM_PROVIDER=mock`), l'appel au
LLM nécessite une connexion Internet active.

**Qui a accès aux quiz que je publie ?**
Tous les étudiants ayant un compte sur l'application peuvent voir et passer
un quiz publié.

**J'ai oublié mon mot de passe, que faire ?**
La version actuelle ne propose pas encore de réinitialisation en libre
service ; contactez votre administrateur système, qui peut recréer un compte
ou (en environnement de développement) modifier directement `data/users.json`.

**Un étudiant peut-il se faire passer pour un autre ?**
Non : le nom affiché dans les résultats est toujours celui du compte
authentifié, jamais une valeur saisie librement.
