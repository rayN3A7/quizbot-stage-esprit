"""
Interface web QuizBot (Streamlit) — refonte visuelle.

Direction artistique : « copie corrigée ». Encre bleu-nuit sur papier froid,
le cramoisi ESPRIT (#C8102E) réservé exclusivement à la main qui corrige
(bonne réponse, filet de marge, action principale) — la même couleur que
celle utilisée dans l'export PDF, pour que l'écran et l'imprimé parlent la
même langue.

Aucune logique métier n'a changé : mêmes appels HTTP, mêmes états de session,
mêmes fonctionnalités (auth, téléversement, génération avec agent de
vérification, publication, export, passage de quiz, correction).

Les blocs visuels sont rendus en HTML maîtrisé (st.markdown + unsafe_allow_html)
plutôt qu'en s'appuyant sur le DOM interne de Streamlit : c'est le seul moyen
fiable d'obtenir un rendu stable d'une version de Streamlit à l'autre. Les
widgets Streamlit (champs, boutons, radios) restent utilisés pour tout ce qui
est interactif, et sont habillés par CSS.
"""
from __future__ import annotations

import os
import sys
from html import escape as esc
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

# `streamlit run frontend/app.py` place déjà frontend/ sur sys.path, mais on
# sécurise le cas où l'app est lancée depuis un autre répertoire de travail.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero import HERO_HEIGHT, HERO_HTML  # noqa: E402
from semantic_map_view import MAP_HEIGHT, build_map_html  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="QuizBot — Assistant pédagogique",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --qb-void:     #05080F;
  --qb-deep:     #0A1020;
  --qb-surface:  #0E1626;
  --qb-raised:   #131D31;
  --qb-line:     rgba(140,180,240,.13);
  --qb-line-hi:  rgba(140,180,240,.28);
  --qb-text:     #E8EEF9;
  --qb-dim:      #8899B8;
  --qb-faint:    #5B6B8A;
  --qb-crimson:  #FF3350;
  --qb-crimson-d:#E01634;
  --qb-crimson-soft: rgba(255,51,80,.11);
  --qb-azure:    #7FB4FF;
  --qb-azure-soft: rgba(127,180,255,.11);
  --qb-valid:    #2EE6A0;
  --qb-valid-soft: rgba(46,230,160,.12);
  --qb-display:  'Space Grotesk', system-ui, sans-serif;
  --qb-body:     'IBM Plex Sans', system-ui, sans-serif;
  --qb-mono:     'IBM Plex Mono', ui-monospace, monospace;
  --qb-r:        16px;
  /* Ombres en couches : liseré, reflet du bord haut, contact, proche, ambiante. */
  --qb-glow:     0 0 0 1px rgba(140,180,240,.07), inset 0 1px 0 rgba(255,255,255,.04),
                 0 1px 2px rgba(0,0,0,.45), 0 8px 18px -10px rgba(0,0,0,.7),
                 0 26px 60px -30px rgba(0,0,0,.9);
  /* Carte soulevée (survol) : contact plus net, ombre portée plus longue et teintée. */
  --qb-lift:     0 0 0 1px rgba(255,51,80,.14), inset 0 1px 0 rgba(255,255,255,.06),
                 0 2px 3px rgba(0,0,0,.5), 0 16px 32px -16px rgba(0,0,0,.85),
                 0 44px 80px -36px rgba(224,22,52,.45);
}

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label, input, textarea {
  font-family: var(--qb-body);
}
.stApp { background: var(--qb-void); color: var(--qb-text); }
/* Le header Streamlit reste dans le flux (il contient le bouton de menu
   mobile et le collapse de la sidebar) : on le rend transparent plutôt que
   de le masquer, sinon il laisse une bande opaque en haut de la page. */
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 3.4rem; padding-bottom: 5rem; max-width: 1200px; }
h1,h2,h3,h4 { font-family: var(--qb-display); color: var(--qb-text); letter-spacing:-.03em; }

/* ---------- champ de profondeur global ---------- */
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(52vw 48vw at 84% -10%, rgba(224,22,52,.10), transparent 60%),
    radial-gradient(46vw 46vw at 4% 106%, rgba(60,120,230,.10), transparent 60%);
  animation: qb-drift 30s ease-in-out infinite alternate;
}
.stApp::after {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.16;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
@keyframes qb-drift {
  0%   { transform: translate3d(0,0,0) scale(1); }
  100% { transform: translate3d(-3%,2.5%,0) scale(1.12); }
}
.block-container, [data-testid="stSidebar"] { position:relative; z-index:1; }

/* ---------- masthead ---------- */
.qb-mast { display:flex; align-items:center; gap:14px; margin-bottom:4px; padding-top:6px; }
.qb-logo {
  width:44px; height:44px; border-radius:13px; flex:none; position:relative;
  background:linear-gradient(150deg,var(--qb-crimson-d),#8E0A1D); color:#fff;
  font-family:var(--qb-display); font-weight:700; font-size:16px; letter-spacing:-.04em;
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 14px 34px -12px rgba(224,22,52,.85), inset 0 1px 0 rgba(255,255,255,.28);
}
.qb-wordmark { font-family:var(--qb-display); font-weight:600; font-size:25px; letter-spacing:-.035em; line-height:1; }
.qb-tagline { font-family:var(--qb-mono); font-size:10px; color:var(--qb-faint); letter-spacing:.22em; text-transform:uppercase; margin-top:6px; }

/* ---------- section ---------- */
.qb-section { display:flex; gap:16px; margin:4px 0 20px; }
.qb-section__rule {
  width:2px; border-radius:2px; flex:none; transform-origin:top;
  background:linear-gradient(180deg,var(--qb-crimson),rgba(255,51,80,0));
  animation:qb-stroke .6s cubic-bezier(.16,1,.3,1) both;
}
@keyframes qb-stroke { from{transform:scaleY(0)} to{transform:scaleY(1)} }
.qb-section__eyebrow {
  font-family:var(--qb-mono); font-size:10px; font-weight:500; letter-spacing:.2em;
  text-transform:uppercase; color:var(--qb-crimson); margin:0 0 6px;
}
.qb-section__title { font-family:var(--qb-display); font-size:23px; font-weight:600; margin:0; letter-spacing:-.03em; }
.qb-section__sub { color:var(--qb-dim); font-size:14px; margin:8px 0 0; max-width:64ch; line-height:1.6; }

/* ---------- surfaces vitrées ---------- */
.qb-card, .qb-stat, .qb-row, .qb-q, .qb-res, .qb-agent, .qb-empty {
  background:linear-gradient(155deg, var(--qb-raised), var(--qb-surface));
  border:1px solid var(--qb-line); box-shadow:var(--qb-glow);
  backdrop-filter:blur(6px);
}
.qb-card { border-radius:var(--qb-r); padding:20px 22px; }
.qb-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(152px,1fr)); gap:12px; }
/* overflow:clip et non hidden sur les cartes : hidden en fait des conteneurs de
   défilement, et les couches internes animées par view() se rattacheraient à la
   carte (qui ne défile pas) au lieu de section.main : elles resteraient figées. */
.qb-stat { border-radius:13px; padding:16px 18px; position:relative; overflow:clip; }
.qb-stat::before {
  content:''; position:absolute; left:0; right:0; top:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--qb-azure),transparent); opacity:.4;
}
.qb-stat__n { font-family:var(--qb-mono); font-size:27px; font-weight:500; line-height:1; letter-spacing:-.04em; color:#fff; }
.qb-stat__l { font-family:var(--qb-mono); font-size:9.5px; color:var(--qb-faint); letter-spacing:.18em; text-transform:uppercase; margin-top:9px; }

.qb-tag {
  display:inline-block; font-family:var(--qb-mono); font-size:9.5px; font-weight:500;
  letter-spacing:.16em; text-transform:uppercase; padding:4px 10px; border-radius:100px;
  background:rgba(140,180,240,.09); color:var(--qb-dim);
  border:1px solid var(--qb-line); margin-right:6px;
}
.qb-tag--crimson { background:var(--qb-crimson-soft); color:var(--qb-crimson); border-color:rgba(255,51,80,.3); }
.qb-tag--valid   { background:var(--qb-valid-soft);   color:var(--qb-valid);   border-color:rgba(46,230,160,.3); }

/* ---------- signature : la carte question en volume ---------- */
.qb-q {
  display:flex; border-radius:var(--qb-r); overflow:clip; margin-bottom:13px;
  position:relative; isolation:isolate; transform-origin:50% 100%;
  transition:box-shadow .3s ease, border-color .3s ease;
}
/* Couche de lumière derrière le contenu : plan « rapide » de la parallaxe. */
.qb-q::before {
  content:''; position:absolute; z-index:0; pointer-events:none;
  top:-35%; right:-12%; width:58%; height:150%;
  background:radial-gradient(closest-side, rgba(127,180,255,.09), transparent);
}
.qb-q:hover { border-color:var(--qb-line-hi); box-shadow:var(--qb-lift); }
.qb-q__margin {
  flex:none; width:62px; border-right:1px solid var(--qb-line);
  background:linear-gradient(180deg, rgba(255,51,80,.10), transparent);
  display:flex; justify-content:center; padding-top:19px;
  position:relative; z-index:1;
}
.qb-q__num { font-family:var(--qb-mono); font-size:13px; font-weight:500; color:var(--qb-crimson); letter-spacing:-.02em; }
.qb-q__body { padding:17px 22px 19px; flex:1; min-width:0; position:relative; z-index:1; }
.qb-q__meta { margin-bottom:11px; }
.qb-q__text { font-size:15.5px; font-weight:500; line-height:1.58; margin:0 0 13px; color:#fff; }
.qb-q__choices { list-style:none; padding:0; margin:0; }
.qb-q__choices li {
  display:flex; align-items:flex-start; gap:11px; padding:9px 13px; border-radius:10px;
  font-size:14px; line-height:1.5; color:var(--qb-dim); border:1px solid transparent;
  margin-bottom:4px; transition:background .2s ease, border-color .2s ease;
}
.qb-q__choices li:hover { background:rgba(140,180,240,.05); }
.qb-q__choices li.is-correct {
  background:var(--qb-crimson-soft); border-color:rgba(255,51,80,.32); color:#fff; font-weight:500;
  box-shadow:inset 0 0 24px -8px rgba(255,51,80,.45);
}
.qb-q__key { font-family:var(--qb-mono); font-size:11px; font-weight:500; flex:none; padding-top:2px; opacity:.65; }
.qb-q__mark { margin-left:auto; color:var(--qb-crimson); font-weight:700; flex:none;
  animation:qb-tick .5s cubic-bezier(.34,1.56,.64,1) .2s both; }
@keyframes qb-tick { from{opacity:0;transform:scale(.3) rotate(-30deg)} to{opacity:1;transform:scale(1) rotate(0)} }
.qb-q__note { margin-top:14px; padding-top:13px; border-top:1px solid var(--qb-line);
  font-size:13.2px; color:var(--qb-dim); line-height:1.65; }
.qb-q__note b { color:var(--qb-text); font-weight:600; }
.qb-q__src { font-family:var(--qb-mono); font-size:11.5px; color:var(--qb-faint); }

/* ---------- rapport de l'agent ---------- */
.qb-agent {
  display:flex; flex-wrap:wrap; align-items:center; gap:26px;
  border-radius:13px; padding:15px 20px; margin-bottom:16px; position:relative; overflow:hidden;
}
.qb-agent::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width:2px;
  background:linear-gradient(180deg,var(--qb-crimson),transparent);
}
.qb-agent__t { font-family:var(--qb-mono); font-size:10px; letter-spacing:.2em; text-transform:uppercase; color:var(--qb-crimson); font-weight:500; }
.qb-agent__i { display:flex; align-items:baseline; gap:8px; }
.qb-agent__n { font-family:var(--qb-mono); font-size:18px; font-weight:500; color:#fff; }
.qb-agent__l { font-size:12px; color:var(--qb-dim); }

/* ---------- anneau de score ---------- */
.qb-score { display:flex; align-items:center; gap:26px; border-radius:var(--qb-r); padding:22px 24px; }
.qb-ring { width:112px; height:112px; border-radius:50%; flex:none; display:flex; align-items:center; justify-content:center;
  animation:qb-pop .7s cubic-bezier(.16,1,.3,1) both; }
@keyframes qb-pop { from{opacity:0;transform:scale(.8) rotate(-14deg)} to{opacity:1;transform:scale(1) rotate(0)} }
.qb-ring__in { width:88px; height:88px; border-radius:50%; background:var(--qb-surface);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  box-shadow:inset 0 0 30px -10px rgba(0,0,0,.9); }
.qb-ring__p { font-family:var(--qb-mono); font-size:24px; font-weight:500; line-height:1; letter-spacing:-.04em; color:#fff; }
.qb-ring__u { font-family:var(--qb-mono); font-size:9px; color:var(--qb-faint); letter-spacing:.2em; margin-top:5px; }
.qb-score__h { font-family:var(--qb-display); font-size:21px; font-weight:600; margin:0 0 5px; }
.qb-score__s { font-family:var(--qb-mono); font-size:13px; color:var(--qb-dim); }

/* ---------- résultats ---------- */
.qb-res { border-radius:13px; padding:15px 18px; margin-bottom:10px; position:relative; overflow:clip;
  transform-origin:0% 50%; transition:box-shadow .3s ease, border-color .25s ease; }
.qb-res:hover { border-color:var(--qb-line-hi); box-shadow:var(--qb-lift); }
.qb-res::before { content:''; position:absolute; left:0; top:0; bottom:0; width:2px; }
.qb-res--ok::before { background:linear-gradient(180deg,var(--qb-valid),transparent); }
.qb-res--ko::before { background:linear-gradient(180deg,var(--qb-crimson),transparent); }
.qb-res__q { font-size:14.5px; font-weight:500; margin:0 0 10px; line-height:1.5; color:#fff; }
.qb-res__l { font-family:var(--qb-mono); font-size:9.5px; letter-spacing:.18em; text-transform:uppercase; color:var(--qb-faint); }
.qb-res__v { font-size:13.8px; margin:3px 0 9px; line-height:1.55; color:var(--qb-dim); }

/* ---------- lignes & vides ---------- */
.qb-row { display:flex; align-items:center; gap:15px; border-radius:13px; padding:14px 17px; margin-bottom:9px;
  overflow:clip; transform-origin:0% 50%; transition:box-shadow .3s ease, border-color .25s ease; }
.qb-row:hover { border-color:var(--qb-line-hi); box-shadow:var(--qb-lift); }
.qb-row__i { width:36px; height:36px; border-radius:10px; background:var(--qb-azure-soft); color:var(--qb-azure);
  display:flex; align-items:center; justify-content:center; flex:none; font-size:15px;
  border:1px solid rgba(127,180,255,.2); }
.qb-row__n { font-weight:600; font-size:14.5px; color:#fff; }
.qb-row__m { font-family:var(--qb-mono); font-size:11px; color:var(--qb-faint); margin-top:4px; }

.qb-empty { border-radius:var(--qb-r); border-style:dashed; padding:40px 28px; text-align:center; }
.qb-empty__t { font-family:var(--qb-display); font-size:17px; font-weight:600; margin:0 0 7px; color:#fff; }
.qb-empty__s { color:var(--qb-dim); font-size:13.8px; margin:0; }

/* ---------- apparition ----------
   Les animations passent par les propriétés individuelles translate/scale, jamais
   par transform : une animation remplie (both) l'emporte sur toute déclaration
   normale, et un transform animé neutralisait les effets de survol.
   Uniquement hors mouvement réduit : sans animation, tout est visible d'emblée. */
/* --i : rang dans un groupe rendu d'un seul bloc (.qb-stack, .qb-grid), d'où un
   décalage d'entrée de 70 ms par élément, plafonné au 13e. Posé ici par
   :nth-child et non en style en ligne : le rendu Markdown de Streamlit retire
   les propriétés personnalisées des attributs style. */
.qb-stack > :nth-child(2),  .qb-grid > :nth-child(2)  { --i:1 }
.qb-stack > :nth-child(3),  .qb-grid > :nth-child(3)  { --i:2 }
.qb-stack > :nth-child(4),  .qb-grid > :nth-child(4)  { --i:3 }
.qb-stack > :nth-child(5),  .qb-grid > :nth-child(5)  { --i:4 }
.qb-stack > :nth-child(6),  .qb-grid > :nth-child(6)  { --i:5 }
.qb-stack > :nth-child(7),  .qb-grid > :nth-child(7)  { --i:6 }
.qb-stack > :nth-child(8),  .qb-grid > :nth-child(8)  { --i:7 }
.qb-stack > :nth-child(9),  .qb-grid > :nth-child(9)  { --i:8 }
.qb-stack > :nth-child(10), .qb-grid > :nth-child(10) { --i:9 }
.qb-stack > :nth-child(11), .qb-grid > :nth-child(11) { --i:10 }
.qb-stack > :nth-child(12), .qb-grid > :nth-child(12) { --i:11 }
.qb-stack > :nth-child(n+13), .qb-grid > :nth-child(n+13) { --i:12 }
@keyframes qb-rise   { from{opacity:0;translate:0 12px} to{opacity:1;translate:0 0} }
@keyframes qb-reveal { from{opacity:0;translate:0 34px} to{opacity:1;translate:0 0} }
@keyframes qb-focus  { from{scale:.965;filter:blur(5px)} to{scale:1;filter:none} }
@media (prefers-reduced-motion: no-preference) {
  /* Repli (sans animation-timeline) : montée à l'affichage. */
  .qb-section,.qb-card,.qb-row,.qb-stat,.qb-q,.qb-res,.qb-empty,.qb-agent {
    animation:qb-rise .55s cubic-bezier(.16,1,.3,1) both;
    animation-delay:calc(var(--i, 0) * 70ms);
  }
  /* Révélation liée au défilement de section.main, plus une mise au point à
     l'affichage pour ce qui est déjà à l'écran : deux jeux de propriétés
     distincts, donc aucun conflit entre les deux animations. */
  @supports (animation-timeline: view()) {
    .qb-section,.qb-card,.qb-row,.qb-stat,.qb-q,.qb-res,.qb-empty,.qb-agent {
      animation:qb-focus .7s cubic-bezier(.16,1,.3,1) both, qb-reveal linear both;
      animation-delay:calc(var(--i, 0) * 70ms), 0s;
      animation-timeline:auto, view();
      animation-range:normal, entry 0% entry 80%;
    }
  }
}

/* ---------- profondeur ----------
   perspective() dans la transformation elle-même : un vrai 3D sans dépendre d'un
   parent (ici des conteneurs internes de Streamlit). Inclinaison fixe au survol,
   en CSS seul, qui laisse le texte sélectionnable. Au défilement, les couches
   internes glissent à des vitesses différentes : numéro et icône lents, lumière
   rapide en sens inverse, contenu immobile. */
@keyframes qb-layer-slow  { from{translate:0 9px}   to{translate:0 -9px} }
@keyframes qb-layer-fast  { from{translate:0 -30px} to{translate:0 30px} }
@keyframes qb-layer-sweep { from{translate:-40% 0}  to{translate:40% 0} }
@media (prefers-reduced-motion: no-preference) {
  .qb-q, .qb-row, .qb-res {
    transition:transform .55s cubic-bezier(.16,1,.3,1), box-shadow .45s cubic-bezier(.16,1,.3,1),
               border-color .3s ease;
  }
  .qb-q:hover   { transform:perspective(1100px) translateY(-4px) rotateX(3.2deg); }
  .qb-row:hover,
  .qb-res:hover { transform:perspective(900px) translateX(4px) rotateY(-3.2deg); }
  @supports (animation-timeline: view()) {
    .qb-q__num, .qb-row__i { animation:qb-layer-slow linear both;  animation-timeline:view(); }
    .qb-q::before          { animation:qb-layer-fast linear both;  animation-timeline:view(); }
    .qb-stat::before       { animation:qb-layer-sweep linear both; animation-timeline:view(); }
  }
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background:var(--qb-deep); border-right:1px solid var(--qb-line); }
[data-testid="stSidebar"] .block-container { padding-top:1.6rem; }
.qb-user { display:flex; align-items:center; gap:12px; padding:14px; border:1px solid var(--qb-line);
  border-radius:13px; background:linear-gradient(155deg,var(--qb-raised),var(--qb-surface)); margin-bottom:16px; }
.qb-user__av { width:38px; height:38px; border-radius:10px; flex:none; color:#fff;
  background:linear-gradient(150deg,var(--qb-crimson-d),#8E0A1D);
  display:flex; align-items:center; justify-content:center; font-family:var(--qb-mono);
  font-size:13px; font-weight:600; box-shadow:0 10px 24px -10px rgba(224,22,52,.8); }
.qb-user__n { font-weight:600; font-size:14px; line-height:1.2; word-break:break-all; color:#fff; }
.qb-user__r { font-family:var(--qb-mono); font-size:9.5px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--qb-crimson); margin-top:4px; }
.qb-side-l { font-family:var(--qb-mono); font-size:9.5px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--qb-faint); margin:20px 0 9px; }

/* ---------- widgets Streamlit ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  font-family:var(--qb-body); font-weight:600; font-size:13.8px; border-radius:11px;
  border:1px solid var(--qb-line); background:var(--qb-raised); color:var(--qb-text);
  padding:.55rem 1.15rem; transition:all .22s cubic-bezier(.16,1,.3,1);
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  border-color:var(--qb-line-hi); background:#182440; transform:translateY(-2px); color:#fff;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:linear-gradient(135deg,var(--qb-crimson),var(--qb-crimson-d));
  border-color:rgba(255,51,80,.5); color:#fff; position:relative; overflow:hidden;
  box-shadow:0 12px 30px -12px rgba(224,22,52,.9);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  box-shadow:0 16px 40px -12px rgba(224,22,52,1); transform:translateY(-2px);
}
.stButton > button[kind="primary"]::after, .stFormSubmitButton > button[kind="primary"]::after {
  content:''; position:absolute; inset:0; transform:translateX(-120%);
  background:linear-gradient(100deg,transparent 20%,rgba(255,255,255,.28) 50%,transparent 80%);
}
.stButton > button[kind="primary"]:hover::after, .stFormSubmitButton > button[kind="primary"]:hover::after {
  animation:qb-sheen .8s ease forwards;
}
@keyframes qb-sheen { to{transform:translateX(120%)} }
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible,
input:focus-visible, textarea:focus-visible {
  outline:2px solid var(--qb-crimson) !important; outline-offset:2px !important;
}

.stTextInput input, .stTextArea textarea, .stNumberInput input {
  border-radius:11px !important; border:1px solid var(--qb-line) !important;
  background:var(--qb-surface) !important; font-size:14px !important; color:var(--qb-text) !important;
  transition:border-color .2s ease, box-shadow .2s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color:var(--qb-crimson) !important; box-shadow:0 0 0 3.5px var(--qb-crimson-soft) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color:var(--qb-faint) !important; }
label, .stTextInput label, .stSelectbox label, .stSlider label, .stTextArea label, .stRadio label {
  font-family:var(--qb-mono) !important; font-size:10px !important; font-weight:500 !important;
  letter-spacing:.16em !important; text-transform:uppercase !important; color:var(--qb-faint) !important;
}
[data-testid="stCheckbox"] label {
  font-family:var(--qb-body) !important; font-size:13.8px !important; letter-spacing:normal !important;
  text-transform:none !important; color:var(--qb-text) !important; font-weight:500 !important;
}

[data-baseweb="select"] > div {
  border-radius:11px !important; border-color:var(--qb-line) !important;
  background:var(--qb-surface) !important; color:var(--qb-text) !important;
}
[data-baseweb="popover"] div, [role="listbox"] { background:var(--qb-raised) !important; color:var(--qb-text) !important; }

[data-testid="stFileUploaderDropzone"] {
  border:1.5px dashed var(--qb-line-hi) !important; border-radius:var(--qb-r) !important;
  background:var(--qb-surface) !important; transition:all .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color:var(--qb-crimson) !important; background:var(--qb-raised) !important; transform:translateY(-2px);
}

/* Streamlit n'expose pas les libellés du téléverseur (toujours en anglais) :
   on masque le texte d'origine et on réécrit l'équivalent français en
   pseudo-éléments. Le bouton conserve son libellé natif accessible pour les
   lecteurs d'écran ; seul l'affichage change. */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
  content:"Glissez un fichier ici";
  display:block; font-family:var(--qb-body); font-size:14.5px; font-weight:600; color:var(--qb-text);
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
  content:"50 Mo maximum · PDF ou PPTX";
  display:block; font-family:var(--qb-mono); font-size:11px; color:var(--qb-faint);
  letter-spacing:.06em; margin-top:5px;
}
[data-testid="stFileUploaderDropzone"] button { font-size:0 !important; }
[data-testid="stFileUploaderDropzone"] button::after {
  content:"Parcourir"; font-size:13.8px; font-family:var(--qb-body); font-weight:600;
}

.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--qb-line); background:transparent; }
.stTabs [data-baseweb="tab"] {
  font-family:var(--qb-body); font-size:13.8px; font-weight:500; color:var(--qb-dim);
  padding:10px 16px; background:transparent; border-radius:9px 9px 0 0; transition:all .2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--qb-text); background:rgba(140,180,240,.05); }
.stTabs [aria-selected="true"] { color:var(--qb-crimson) !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--qb-crimson); }

[data-testid="stExpander"] {
  border:1px solid var(--qb-line) !important; border-radius:13px !important;
  background:linear-gradient(155deg,var(--qb-raised),var(--qb-surface)) !important; box-shadow:none !important;
}
[data-testid="stExpander"] summary { font-size:14px !important; font-weight:500 !important; color:var(--qb-text) !important; }
hr { border-color:var(--qb-line); }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--qb-crimson) !important; }
[data-testid="stSpinner"] > div { border-top-color:var(--qb-crimson) !important; }
.stAlert { background:var(--qb-raised) !important; border:1px solid var(--qb-line) !important;
  border-radius:12px !important; color:var(--qb-text) !important; }

@media (prefers-reduced-motion: reduce) {
  /* `*` seul ne cible pas les pseudo-éléments (fond animé, reflet des boutons). */
  *, *::before, *::after { transition:none !important; animation:none !important; }
  .stButton > button:hover, .qb-q:hover, .qb-row:hover, .qb-res:hover { transform:none; }
}
@media (max-width: 640px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .qb-q__margin { width:44px; }
  .qb-score { flex-direction:column; align-items:flex-start; gap:16px; }
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Composants de présentation
# --------------------------------------------------------------------------- #

def masthead(tagline: str = "Assistant pédagogique — ESPRIT") -> None:
    st.markdown(
        f'<div class="qb-mast"><div class="qb-logo">QB</div><div>'
        f'<div class="qb-wordmark">QuizBot</div>'
        f'<div class="qb-tagline">{esc(tagline)}</div></div></div>',
        unsafe_allow_html=True,
    )


def section(eyebrow: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="qb-section__sub">{esc(sub)}</p>' if sub else ""
    st.markdown(
        f'<div class="qb-section"><span class="qb-section__rule"></span><div>'
        f'<p class="qb-section__eyebrow">{esc(eyebrow)}</p>'
        f'<h2 class="qb-section__title">{esc(title)}</h2>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def _txt(value) -> str:
    """Texte échappé tenant sur une seule ligne de source. Dans un bloc HTML, une
    ligne vide termine le bloc pour le parseur Markdown : la suite sortirait de sa
    carte, et d'un groupe rendu d'un seul tenant, tout ce qui la suit."""
    return esc(str(value)).replace("\r\n", "\n").replace("\n", "<br>")


def stack(items) -> None:
    """Rend un groupe en un seul bloc : ses éléments entrent l'un après l'autre,
    décalés selon leur rang (voir --i dans le CSS)."""
    st.markdown(f'<div class="qb-stack">{"".join(items)}</div>', unsafe_allow_html=True)


def stats(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="qb-stat"><div class="qb-stat__n">{_txt(n)}</div>'
        f'<div class="qb-stat__l">{_txt(label)}</div></div>'
        for n, label in items
    )
    st.markdown(f'<div class="qb-grid">{cells}</div>', unsafe_allow_html=True)


def empty_state(title: str, sub: str) -> None:
    st.markdown(
        f'<div class="qb-empty"><p class="qb-empty__t">{esc(title)}</p>'
        f'<p class="qb-empty__s">{esc(sub)}</p></div>',
        unsafe_allow_html=True,
    )


def row_html(icon: str, name: str, meta: str) -> str:
    return (
        f'<div class="qb-row"><div class="qb-row__i">{_txt(icon)}</div><div>'
        f'<div class="qb-row__n">{_txt(name)}</div>'
        f'<div class="qb-row__m">{_txt(meta)}</div></div></div>'
    )


def question_card_html(index: int, q: dict) -> str:
    """Élément signature : la question rendue comme une ligne de copie corrigée."""
    tags = (
        f'<span class="qb-tag qb-tag--crimson">{_txt(q["type"])}</span>'
        f'<span class="qb-tag">{_txt(q.get("difficulty", ""))}</span>'
    )
    if q.get("theme"):
        tags += f'<span class="qb-tag">{_txt(q["theme"])}</span>'

    if q["type"] == "qcm" and q.get("choices"):
        lis = ""
        for k, choice in enumerate(q["choices"]):
            ok = k == q.get("correct_choice_index")
            mark = '<span class="qb-q__mark">✓</span>' if ok else ""
            lis += (
                f'<li class="{"is-correct" if ok else ""}">'
                f'<span class="qb-q__key">{chr(65 + k)}</span>'
                f'<span>{_txt(choice)}</span>{mark}</li>'
            )
        answer_html = f'<ul class="qb-q__choices">{lis}</ul>'
    else:
        answer_html = (
            f'<ul class="qb-q__choices"><li class="is-correct">'
            f'<span class="qb-q__key">RÉP</span>'
            f'<span>{_txt(q.get("reference_answer", ""))}</span>'
            f'<span class="qb-q__mark">✓</span></li></ul>'
        )

    note = ""
    if q.get("explanation"):
        note += f'<div class="qb-q__note"><b>Pourquoi</b> — {_txt(q["explanation"])}</div>'
    if q.get("source_excerpt"):
        excerpt = str(q["source_excerpt"])[:180]
        note += (
            f'<div class="qb-q__note"><b>Extrait du cours</b><br>'
            f'<span class="qb-q__src">{_txt(excerpt)}…</span></div>'
        )

    return (
        f'<article class="qb-q"><div class="qb-q__margin">'
        f'<span class="qb-q__num">{index:02d}</span></div>'
        f'<div class="qb-q__body"><div class="qb-q__meta">{tags}</div>'
        f'<p class="qb-q__text">{_txt(q["question"])}</p>'
        f'{answer_html}{note}</div></article>'
    )


def agent_report(report: dict) -> None:
    items = [
        (report.get("generated", 0), "générées"),
        (report.get("verified_ok_first_try", 0), "validées d'emblée"),
        (report.get("regenerated", 0), "régénérées"),
        (report.get("dropped", 0), "écartées"),
    ]
    cells = "".join(
        f'<div class="qb-agent__i"><span class="qb-agent__n">{n}</span>'
        f'<span class="qb-agent__l">{esc(label)}</span></div>'
        for n, label in items
    )
    st.markdown(
        f'<div class="qb-agent"><span class="qb-agent__t">Relecture de l\'agent</span>{cells}</div>',
        unsafe_allow_html=True,
    )


def score_ring(percentage: float, score: float, max_score: float) -> None:
    pct = max(0.0, min(float(percentage), 100.0))
    color = "var(--qb-valid)" if pct >= 50 else "var(--qb-crimson)"
    verdict = "Bien joué." if pct >= 75 else ("Presque." if pct >= 50 else "À retravailler.")
    st.markdown(
        f'<div class="qb-card qb-score">'
        f'<div class="qb-ring" style="background:conic-gradient({color} {pct}%, var(--qb-line) 0);">'
        f'<div class="qb-ring__in"><div class="qb-ring__p">{pct:.0f}%</div>'
        f'<div class="qb-ring__u">SCORE</div></div></div>'
        f'<div><p class="qb-score__h">{esc(verdict)}</p>'
        f'<p class="qb-score__s">{score} / {max_score} points</p></div></div>',
        unsafe_allow_html=True,
    )


def result_row_html(g: dict) -> str:
    ok = bool(g["correct"])
    return (
        f'<div class="qb-res qb-res--{"ok" if ok else "ko"}">'
        f'<p class="qb-res__q">{_txt(g["question"])}</p>'
        f'<div class="qb-res__l">Votre réponse</div>'
        f'<div class="qb-res__v">{_txt(str(g["student_answer"]) or "— (vide)")}</div>'
        f'<div class="qb-res__l">Réponse attendue</div>'
        f'<div class="qb-res__v">{_txt(g["correct_answer"])}</div>'
        + (f'<div class="qb-q__note">{_txt(g["explanation"])}</div>' if g.get("explanation") else "")
        + '</div>'
    )


# --------------------------------------------------------------------------- #
# Helpers HTTP (inchangés)
# --------------------------------------------------------------------------- #

def _auth_headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path, **kwargs):
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    r = requests.get(f"{API_BASE_URL}{path}", headers=headers, **kwargs)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.json()


def api_post(path, **kwargs):
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    r = requests.post(f"{API_BASE_URL}{path}", headers=headers, **kwargs)
    _raise_for_auth(r)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def _raise_for_auth(response: requests.Response) -> None:
    if response.status_code == 401:
        _expire_session()


def _expire_session() -> None:
    st.session_state.pop("token", None)
    st.session_state.pop("user", None)
    st.error("Votre session a expiré. Reconnectez-vous pour continuer.")
    st.rerun()


class _SessionExpired(Exception):
    pass


# Les fonctions en cache ne doivent appeler aucune commande Streamlit : un 401
# remonte en exception, et le code appelant déconnecte l'utilisateur.
#
# Clé de cache : (quiz, format, publié). Le jeton n'en fait PAS partie (préfixe
# « _ », ignoré par st.cache_data) : le cache est partagé entre toutes les
# sessions de professeurs, et une reconnexion ne refait pas les exports.
# C'est sûr uniquement parce qu'aujourd'hui tout professeur voit tous les quiz
# (GET /quizzes renvoie tout au rôle professeur ; les exports lui sont réservés).
# Si une visibilité par professeur est ajoutée, cette clé devient une FUITE : un
# professeur recevrait l'export mis en cache par un autre sans que l'API ne
# vérifie ses droits. Il faudra alors remettre l'identité du professeur dans la clé.
# « published » est dans la clé car l'export JSON contient ce drapeau ; rien
# d'autre ne change après la génération, d'où l'absence d'expiration.
@st.cache_data(max_entries=512, show_spinner=False)
def _fetch_export(quiz_id: str, kind: str, published: bool, _token: str) -> bytes:
    r = requests.get(f"{API_BASE_URL}/quizzes/{quiz_id}/export/{kind}",
                     headers={"Authorization": f"Bearer {_token}"})
    if r.status_code == 401:
        raise _SessionExpired
    r.raise_for_status()
    return r.content


def export_button(quiz: dict, kind: str, mime: str) -> None:
    """Téléchargement en un clic : l'export est récupéré à l'affichage de
    « Mes quiz » puis gardé en cache. Un export en échec n'affiche son erreur
    que pour ce quiz ; le reste de l'onglet s'affiche normalement."""
    label = kind.upper()
    try:
        data = _fetch_export(quiz["id"], kind, quiz["published"], st.session_state.get("token", ""))
    except _SessionExpired:
        _expire_session()
    except requests.RequestException as e:
        st.error(f"Export {label} indisponible : {e}")
        return
    st.download_button(f"Télécharger le {label}", data=data, file_name=f"{quiz['title']}.{kind}",
                       mime=mime, key=f"{kind}_{quiz['id']}")


@st.cache_data(ttl=60, max_entries=128, show_spinner=False)
def _fetch_json(path: str, token: str, params: tuple):
    r = requests.get(f"{API_BASE_URL}{path}", headers={"Authorization": f"Bearer {token}"},
                     params=dict(params))
    if r.status_code == 401:
        raise _SessionExpired
    r.raise_for_status()
    return r.json()


def api_get_cached(path: str, **params):
    """Lecture partagée entre les reruns (60 s, par utilisateur). Toute écriture
    appelle invalidate_cache() pour que l'affichage suivant la reflète."""
    try:
        return _fetch_json(path, st.session_state.get("token", ""), tuple(sorted(params.items())))
    except _SessionExpired:
        _expire_session()


def invalidate_cache() -> None:
    _fetch_json.clear()


def _flash(key: str, message: str) -> None:
    st.session_state[f"flash_{key}"] = message


def _show_flash(key: str) -> None:
    if message := st.session_state.pop(f"flash_{key}", None):
        st.success(message)


# --------------------------------------------------------------------------- #
# Vérification de la connexion au backend
# --------------------------------------------------------------------------- #

try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
except Exception:
    masthead()
    st.markdown(
        f'<div class="qb-empty"><p class="qb-empty__t">Le serveur ne répond pas</p>'
        f'<p class="qb-empty__s">Aucune réponse sur {esc(API_BASE_URL)}. '
        f'Lancez <code>uvicorn backend.main:app --port 8000</code>, puis rechargez cette page.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Connexion / inscription
# --------------------------------------------------------------------------- #

def login_register_screen():
    # Le hero passe par un composant iframe : c'est le seul contexte où
    # Streamlit exécute du JavaScript, donc le seul où le parallaxe est possible.
    components.html(HERO_HTML, height=HERO_HEIGHT, scrolling=False)

    left, mid, right = st.columns([1, 1.35, 1])
    with mid:
        st.markdown(
            f'<div style="text-align:center;margin:6px 0 18px;">'
            f'<span class="qb-tag qb-tag--valid">moteur {esc(health["llm_provider"])}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Se connecter", "Créer un compte"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur")
                password = st.text_input("Mot de passe", type="password")
                submitted = st.form_submit_button("Se connecter", type="primary", use_container_width=True)

            if submitted:
                try:
                    r = requests.post(
                        f"{API_BASE_URL}/auth/login",
                        data={"username": username, "password": password},
                    )
                    if r.status_code != 200:
                        st.error(r.json().get("detail", "Nom d'utilisateur ou mot de passe incorrect."))
                    else:
                        data = r.json()
                        st.session_state["token"] = data["access_token"]
                        st.session_state["user"] = {"username": data["username"], "role": data["role"]}
                        st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Le serveur est injoignable : {e}")

        with tab_register:
            with st.form("register_form"):
                reg_username = st.text_input("Nom d'utilisateur", key="reg_username")
                reg_full_name = st.text_input("Nom complet", key="reg_full_name")
                reg_password = st.text_input("Mot de passe (6 caractères minimum)",
                                             type="password", key="reg_password")
                reg_role = st.radio("Vous êtes", ["Étudiant", "Professeur"],
                                    key="reg_role", horizontal=True)
                reg_professor_code = ""
                if reg_role == "Professeur":
                    reg_professor_code = st.text_input(
                        "Code d'inscription professeur",
                        type="password", key="reg_professor_code",
                        help="Fourni par votre administrateur.",
                    )
                reg_submitted = st.form_submit_button("Créer mon compte", type="primary",
                                                      use_container_width=True)

            if reg_submitted:
                role_value = "professeur" if reg_role == "Professeur" else "etudiant"
                try:
                    r = requests.post(f"{API_BASE_URL}/auth/register", json={
                        "username": reg_username,
                        "password": reg_password,
                        "full_name": reg_full_name,
                        "role": role_value,
                        "professor_code": reg_professor_code or None,
                    })
                    if r.status_code != 201:
                        st.error(r.json().get("detail", "La création du compte a échoué."))
                    else:
                        st.success("Compte créé. Connectez-vous dans l'onglet « Se connecter ».")
                except requests.exceptions.RequestException as e:
                    st.error(f"Le serveur est injoignable : {e}")


# --------------------------------------------------------------------------- #
# Espace Enseignant
# --------------------------------------------------------------------------- #

def teacher_space():
    masthead("Espace enseignant")
    st.write("")

    tab_upload, tab_generate, tab_manage, tab_map = st.tabs(
        ["Téléverser un cours", "Générer un quiz", "Mes quiz", "Carte du cours"]
    )

    # Chaque onglet est un fragment : un widget ne relance que son onglet, pas
    # la page entière. Une écriture relance toute l'app (st.rerun) pour que les
    # autres onglets la voient ; le message de succès survit via _flash.

    # --- 1. Téléversement ---------------------------------------------------
    @st.fragment
    def upload_tab():
        section("Étape 1", "Téléverser un support de cours",
                "Le document est découpé en segments puis vectorisé : c'est ce qui "
                "permet de retrouver les passages pertinents au moment de générer un quiz.")

        uploaded_file = st.file_uploader("Fichier de cours (PDF ou PPTX)", type=["pdf", "pptx"])
        if uploaded_file and st.button("Téléverser et indexer", type="primary"):
            with st.spinner("Extraction, découpage et vectorisation…"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    doc = api_post("/documents/upload", files=files)
                except Exception as e:
                    st.error(f"L'indexation a échoué : {e}")
                else:
                    invalidate_cache()
                    st.session_state["last_document_id"] = doc["id"]
                    _flash("upload", f"Document indexé — {doc['num_chunks']} segments créés.")
                    st.rerun()
        _show_flash("upload")

        st.write("")
        docs = api_get_cached("/documents")
        section("Bibliothèque", "Documents indexés",
                "" if docs else "Rien n'est encore indexé.")
        if docs:
            stats([
                (len(docs), "documents"),
                (sum(d["num_chunks"] for d in docs), "segments"),
            ])
            st.write("")
            stack(row_html("📄", d["filename"],
                           f"{d['num_chunks']} segments · déposé par {d['uploaded_by']}")
                  for d in docs)
        else:
            empty_state("Aucun document pour l'instant",
                        "Déposez un PDF ou un PPTX ci-dessus pour commencer.")

    # --- 2. Génération ------------------------------------------------------
    @st.fragment
    def generate_tab():
        docs = api_get_cached("/documents")
        if not docs:
            section("Étape 2", "Générer un quiz")
            empty_state("Il faut d'abord un document",
                        "Téléversez un support de cours dans l'onglet précédent.")
        else:
            section("Étape 2", "Configurer et générer",
                    "Les questions sont écrites à partir des passages les plus pertinents "
                    "du document choisi.")

            doc_options = {f"{d['filename']} — {d['num_chunks']} segments": d["id"] for d in docs}
            doc_label = st.selectbox("Document source", list(doc_options.keys()))
            document_id = doc_options[doc_label]

            col1, col2, col3 = st.columns(3)
            with col1:
                title = st.text_input("Titre du quiz", value="Quiz — Chapitre 1")
                num_questions = st.slider("Nombre de questions", 1, 20, 5)
            with col2:
                question_type = st.selectbox("Type de questions", ["mélange", "qcm", "ouverte"])
                difficulty = st.selectbox("Difficulté", ["facile", "moyen", "difficile"], index=1)
            with col3:
                themes_input = st.text_area("Thèmes à privilégier",
                                            placeholder="Un thème par ligne (facultatif)")

            use_verification_agent = st.checkbox(
                "Faire relire les questions par l'agent de vérification",
                help=(
                    "Chaque question est repassée au modèle, qui vérifie qu'elle est "
                    "répondable à partir du cours et que la réponse indiquée est juste. "
                    "Les questions rejetées sont régénérées (2 tentatives), puis écartées. "
                    "Plus lent, mais nettement plus fiable."
                ),
            )

            if st.button("Générer le quiz", type="primary"):
                themes = [t.strip() for t in themes_input.splitlines() if t.strip()] or None
                config = {
                    "document_id": document_id,
                    "title": title,
                    "num_questions": num_questions,
                    "question_type": question_type,
                    "difficulty": difficulty,
                    "themes": themes,
                    "use_verification_agent": use_verification_agent,
                }
                spinner_text = (
                    "Rédaction puis relecture de chaque question…"
                    if use_verification_agent else
                    "Recherche des passages puis rédaction des questions…"
                )
                with st.spinner(spinner_text):
                    try:
                        quiz = api_post("/quizzes/generate", json=config)
                    except Exception as e:
                        st.error(f"La génération a échoué : {e}")
                    else:
                        invalidate_cache()
                        st.session_state["generated_quiz"] = quiz
                        st.rerun()

            if "generated_quiz" in st.session_state:
                quiz = st.session_state["generated_quiz"]
                st.write("")
                section("Relecture", quiz["title"],
                        "Vérifiez les questions avant de publier — les étudiants ne verront "
                        "que ce qui est publié.")

                if quiz.get("agent_report"):
                    agent_report(quiz["agent_report"])

                stack(question_card_html(n, q) for n, q in enumerate(quiz["questions"], start=1))

                st.write("")
                colA, colB = st.columns([1, 2])
                with colA:
                    if st.button("Publier pour les étudiants", type="primary"):
                        api_post(f"/quizzes/{quiz['id']}/publish")
                        invalidate_cache()
                        _flash("publish", "Quiz publié.")
                        st.rerun()
                    _show_flash("publish")
                with colB:
                    st.markdown(
                        f'<div class="qb-row__m" style="padding-top:10px;">'
                        f'identifiant {esc(quiz["id"])}</div>',
                        unsafe_allow_html=True,
                    )

    # --- 3. Gestion ---------------------------------------------------------
    @st.fragment
    def manage_tab():
        quizzes = api_get_cached("/quizzes")
        section("Bibliothèque", "Mes quiz",
                "" if quizzes else "Vous n'avez encore rien généré.")

        if not quizzes:
            empty_state("Aucun quiz pour l'instant",
                        "Générez votre premier questionnaire dans l'onglet « Générer un quiz ».")
        else:
            published = sum(1 for q in quizzes if q["published"])
            stats([
                (len(quizzes), "quiz"),
                (published, "publiés"),
                (len(quizzes) - published, "brouillons"),
                (sum(len(q["questions"]) for q in quizzes), "questions"),
            ])
            st.write("")

            for quiz in quizzes:
                state = ("qb-tag--valid", "publié") if quiz["published"] else ("", "brouillon")
                header = f"{quiz['title']}  ·  {len(quiz['questions'])} questions  ·  {state[1]}"
                with st.expander(header):
                    st.markdown(
                        f'<div class="qb-row__m" style="margin-bottom:12px;">'
                        f'source {esc(quiz["document_name"])} · '
                        f'auteur {esc(quiz["created_by"])} · '
                        f'identifiant {esc(quiz["id"])}</div>',
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if not quiz["published"] and st.button("Publier", key=f"pub_{quiz['id']}"):
                            api_post(f"/quizzes/{quiz['id']}/publish")
                            invalidate_cache()
                            st.rerun()
                    with c2:
                        export_button(quiz, "pdf", "application/pdf")
                    with c3:
                        export_button(quiz, "json", "application/json")


    # --- 4. Carte sémantique ------------------------------------------------
    @st.fragment
    def map_tab():
        docs = api_get_cached("/documents")
        section("Analyse", "Carte sémantique du cours",
                "Chaque point est un segment du document, positionné selon son sens : "
                "deux passages proches traitent de sujets proches. Les amas révèlent "
                "les grandes sections conceptuelles du support.")

        if not docs:
            empty_state("Aucun document à cartographier",
                        "Téléversez un support de cours pour visualiser sa structure.")
        else:
            map_options = {f"{d['filename']} — {d['num_chunks']} segments": d["id"] for d in docs}
            map_label = st.selectbox("Document", list(map_options.keys()), key="map_doc")
            show_perf = st.checkbox(
                "Superposer les résultats des étudiants",
                help="Colore chaque segment selon le score moyen obtenu sur les questions "
                     "qui en sont issues : les zones rouges signalent les parties du cours "
                     "les moins bien comprises.",
            )

            with st.spinner("Projection des segments en deux dimensions…"):
                try:
                    payload = api_get_cached(
                        f"/documents/{map_options[map_label]}/map", with_performance=show_perf,
                    )
                except Exception as e:
                    payload = None
                    st.error(f"La carte n'a pas pu être construite : {e}")

            if payload:
                components.html(
                    build_map_html(payload, show_perf), height=MAP_HEIGHT, scrolling=False
                )
                if show_perf:
                    covered = payload.get("covered_points", 0)
                    total = payload.get("num_points", 0)
                    st.markdown(
                        f'<div class="qb-row__m">{covered} segment(s) sur {total} '
                        f'ont déjà été évalués par au moins une question.</div>',
                        unsafe_allow_html=True,
                    )

    for tab, render in ((tab_upload, upload_tab), (tab_generate, generate_tab),
                        (tab_manage, manage_tab), (tab_map, map_tab)):
        with tab:
            render()


# --------------------------------------------------------------------------- #
# Espace Étudiant
# --------------------------------------------------------------------------- #

def student_space():
    masthead("Espace étudiant")
    st.write("")

    quizzes = api_get("/quizzes", params={"published_only": True})
    if not quizzes:
        section("Auto-évaluation", "Quiz disponibles")
        empty_state("Aucun quiz publié pour le moment",
                    "Vos enseignants n'ont encore rien mis en ligne. Revenez plus tard.")
        return

    section("Auto-évaluation", "Choisir un quiz",
            "Répondez sans le cours sous les yeux : la correction arrive juste après.")

    quiz_options = {f"{q['title']} — {len(q['questions'])} questions": q["id"] for q in quizzes}
    quiz_label = st.selectbox("Quiz", list(quiz_options.keys()))
    quiz_id = quiz_options[quiz_label]
    quiz = api_get(f"/quizzes/{quiz_id}")

    st.write("")
    answers = {}
    with st.form("quiz_form"):
        for i, q in enumerate(quiz["questions"], start=1):
            st.markdown(
                f'<div style="margin:18px 0 6px;">'
                f'<span class="qb-tag qb-tag--crimson">question {i:02d}</span>'
                f'<p class="qb-q__text" style="margin-top:9px;">{_txt(q["question"])}</p></div>',
                unsafe_allow_html=True,
            )
            if q["type"] == "qcm" and q.get("choices"):
                # Le widget renvoie l'indice, jamais le texte : aucune recherche
                # textuelle, et rien de présélectionné. Une question sautée part
                # en "" (comptée fausse) au lieu du choix A.
                picked = st.radio(
                    "Votre réponse", range(len(q["choices"])),
                    format_func=q["choices"].__getitem__, index=None,
                    key=f"answer_{q['id']}", label_visibility="collapsed",
                )
                answers[q["id"]] = "" if picked is None else str(picked)
            else:
                answers[q["id"]] = st.text_area(
                    "Votre réponse", key=f"answer_{q['id']}",
                    label_visibility="collapsed", placeholder="Rédigez votre réponse…",
                )

        st.write("")
        submitted = st.form_submit_button("Remettre ma copie", type="primary")

    if submitted:
        submission = {
            "quiz_id": quiz_id,
            "answers": [{"question_id": qid, "answer": ans} for qid, ans in answers.items()],
        }
        try:
            result = api_post(f"/quizzes/{quiz_id}/submit", json=submission)
        except Exception as e:
            st.error(f"La remise a échoué : {e}")
            return

        st.write("")
        section("Correction", "Votre résultat")
        score_ring(result["percentage"], result["total_score"], result["max_score"])

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="qb-side-l">Par thème</div>', unsafe_allow_html=True)
            if result["stats_by_theme"]:
                stats([(f"{v:.0f}%", k) for k, v in result["stats_by_theme"].items()])
            else:
                st.caption("Pas de thème renseigné.")
        with col2:
            st.markdown('<div class="qb-side-l">Par type de question</div>', unsafe_allow_html=True)
            stats([(f"{v:.0f}%", k) for k, v in result["stats_by_type"].items()])

        st.write("")
        section("Détail", "Question par question")
        stack(result_row_html(g) for g in result["graded_answers"])


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #

if "token" not in st.session_state:
    login_register_screen()
else:
    user = st.session_state["user"]
    initials = (user["username"][:2] or "??").upper()

    with st.sidebar:
        st.markdown(
            f'<div class="qb-user"><div class="qb-user__av">{esc(initials)}</div><div>'
            f'<div class="qb-user__n">{esc(user["username"])}</div>'
            f'<div class="qb-user__r">{esc(user["role"])}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="qb-side-l">Moteur</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="qb-tag qb-tag--valid">{esc(health["llm_provider"])}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="qb-side-l">Session</div>', unsafe_allow_html=True)
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.pop("token", None)
            st.session_state.pop("user", None)
            st.rerun()

    if user["role"] == "professeur":
        teacher_space()
    else:
        student_space()
