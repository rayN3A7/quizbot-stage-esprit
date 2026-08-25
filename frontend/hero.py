"""
Hero 3D de QuizBot — l'espace vectoriel.

Direction : ce que fait réellement l'application, montré littéralement. Chaque
segment de cours devient un point dans un nuage d'embeddings ; générer un quiz
revient à faire remonter les plus pertinents. Le hero met cette scène en volume
plutôt que d'empiler des dégradés décoratifs.

Trois techniques de profondeur, cumulées :
  1. Nuage de points projeté en 3D sur <canvas> — chaque point a un x/y/z réel,
     tourne autour de l'axe Y, et sa taille comme son opacité découlent de la
     perspective (focale / (focale + z)). Pas une imitation 2D de profondeur.
  2. Plans de document en CSS 3D — perspective + preserve-3d + translateZ,
     inclinés en rotateX/rotateY selon le pointeur : le volume entier pivote,
     donc les plans se déforment correctement.
  3. Parallaxe d'inertie sur le contenu — le texte dérive à contre-sens et plus
     lentement, ce qui achève de séparer les plans.

Rendu via st.components.v1.html : Streamlit retire les <script> des blocs
st.markdown, l'iframe d'un composant est le seul contexte où le JS s'exécute.
"""

HERO_HEIGHT = 496

HERO_HTML = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;450&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:transparent;overflow:hidden}

  #stage{
    position:relative;width:100%;height:480px;border-radius:22px;overflow:hidden;
    background:radial-gradient(130% 120% at 78% -20%, #101C36 0%, #070C18 52%, #04070E 100%);
    font-family:'IBM Plex Sans',system-ui,sans-serif;
    perspective:1100px;perspective-origin:50% 42%;
  }

  /* ---------- 1. nuage d'embeddings ---------- */
  #cloud{position:absolute;inset:0;width:100%;height:100%;display:block}

  /* ---------- 2. plans de document en volume ---------- */
  #space{
    position:absolute;inset:0;transform-style:preserve-3d;will-change:transform;
  }
  .plane{
    position:absolute;border-radius:14px;
    border:1px solid rgba(140,190,255,.20);
    background:linear-gradient(152deg,rgba(120,170,255,.10),rgba(120,170,255,.02));
    box-shadow:0 30px 70px -30px rgba(0,0,0,.95), inset 0 1px 0 rgba(255,255,255,.10);
    padding:14px 15px;backdrop-filter:blur(3px);
  }
  .plane b{
    display:block;font-family:'IBM Plex Mono';font-size:8.5px;letter-spacing:.2em;
    color:rgba(150,195,255,.55);margin-bottom:9px;font-weight:400;
  }
  .plane i{display:block;height:4px;border-radius:2px;background:rgba(160,200,255,.22);margin-bottom:6px}
  .plane i:nth-of-type(2){width:88%}
  .plane i:nth-of-type(3){width:96%}
  .plane i:nth-of-type(4){width:71%}

  /* le plan « retenu » : celui que le retrieval a fait remonter */
  .plane--hit{
    border-color:rgba(224,22,52,.55);
    background:linear-gradient(152deg,rgba(224,22,52,.16),rgba(224,22,52,.03));
    box-shadow:0 30px 70px -28px rgba(224,22,52,.55), inset 0 1px 0 rgba(255,255,255,.14);
  }
  .plane--hit b{color:rgba(255,120,140,.85)}
  .plane--hit i{background:rgba(255,140,160,.32)}

  .p1{width:158px;left:4%;   top:24%;    transform:translateZ(-180px) rotateY(17deg)  rotateX(5deg)}
  .p2{width:132px;left:14%;  bottom:13%; transform:translateZ(-320px) rotateY(11deg)  rotateX(-6deg)}
  .p3{width:170px;right:6%;  top:19%;    transform:translateZ(-90px)  rotateY(-19deg) rotateX(4deg)}
  .p4{width:140px;right:17%; bottom:10%; transform:translateZ(-260px) rotateY(-13deg) rotateX(-5deg)}

  /* ---------- 3. contenu ---------- */
  #content{
    position:absolute;inset:0;z-index:5;display:flex;flex-direction:column;justify-content:center;
    padding:0 clamp(26px,6.5vw,74px);will-change:transform;
  }
  .brand{display:flex;align-items:center;gap:13px;margin-bottom:26px}
  .mark{
    width:46px;height:46px;border-radius:13px;flex:none;position:relative;
    background:linear-gradient(150deg,#E01634,#8E0A1D);color:#fff;
    font-family:'Space Grotesk';font-weight:700;font-size:17px;letter-spacing:-.04em;
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 16px 38px -12px rgba(224,22,52,.9), inset 0 1px 0 rgba(255,255,255,.3);
  }
  .mark::after{
    content:'';position:absolute;inset:-3px;border-radius:16px;
    border:1px solid rgba(224,22,52,.35);animation:halo 3.4s ease-in-out infinite;
  }
  @keyframes halo{0%,100%{opacity:.25;transform:scale(1)}50%{opacity:.75;transform:scale(1.07)}}
  .bname{font-family:'Space Grotesk';font-weight:600;font-size:19px;color:#fff;letter-spacing:-.02em}
  .beyebrow{font-family:'IBM Plex Mono';font-size:9.5px;letter-spacing:.24em;color:rgba(150,190,255,.5);margin-top:4px}

  h1{
    font-family:'Space Grotesk';font-weight:600;color:#fff;
    font-size:clamp(29px,4.6vw,50px);line-height:1.04;letter-spacing:-.04em;max-width:16ch;
  }
  .grad{
    background:linear-gradient(96deg,#FF3350 0%,#FF7A8C 38%,#7FB4FF 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent;
  }
  .lede{color:rgba(200,218,245,.58);font-size:14.5px;line-height:1.66;max-width:44ch;margin-top:26px}

  .metrics{display:flex;gap:30px;margin-top:30px;flex-wrap:wrap}
  .metric b{
    display:block;font-family:'IBM Plex Mono';font-size:19px;font-weight:500;color:#fff;
    letter-spacing:-.03em;line-height:1;
  }
  .metric span{
    display:block;font-family:'IBM Plex Mono';font-size:9.5px;letter-spacing:.17em;
    color:rgba(150,190,255,.5);margin-top:7px;
  }
  .metric b em{font-style:normal;color:#FF5C74}

  /* ---------- entrée ---------- */
  .rise{opacity:0;transform:translateY(26px);animation:rise .9s cubic-bezier(.16,1,.3,1) forwards}
  .d1{animation-delay:.08s}.d2{animation-delay:.24s}.d3{animation-delay:.42s}.d4{animation-delay:.60s}
  @keyframes rise{to{opacity:1;transform:translateY(0)}}
  .plane{opacity:0;animation:planein 1.1s cubic-bezier(.16,1,.3,1) forwards}
  .p1{animation-delay:.30s}.p2{animation-delay:.44s}.p3{animation-delay:.38s}.p4{animation-delay:.52s}
  @keyframes planein{to{opacity:1}}

  /* ---------- vignette + grain ---------- */
  #stage::before{
    content:'';position:absolute;inset:0;z-index:4;pointer-events:none;
    background:radial-gradient(78% 78% at 50% 44%,transparent 42%,rgba(2,5,12,.72) 100%);
  }
  #stage::after{
    content:'';position:absolute;inset:0;z-index:6;pointer-events:none;opacity:.20;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='150'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  @media(prefers-reduced-motion:reduce){
    .rise,.plane{opacity:1;animation:none}
    .rise{transform:none}
    .mark::after{animation:none}
  }
  @media(max-width:620px){
    .plane{display:none}
    .metrics{gap:20px}
    .lede{margin-top:18px;font-size:13.5px}
  }
</style>
</head>
<body>
<div id="stage">
  <canvas id="cloud"></canvas>

  <div id="space">
    <div class="plane p1"><b>SEGMENT 041</b><i></i><i></i><i></i><i></i></div>
    <div class="plane p2"><b>SEGMENT 108</b><i></i><i></i><i></i><i></i></div>
    <div class="plane p3 plane--hit"><b>RETENU 0.91</b><i></i><i></i><i></i><i></i></div>
    <div class="plane p4"><b>SEGMENT 219</b><i></i><i></i><i></i><i></i></div>
  </div>

  <div id="content">
    <div class="brand rise d1">
      <div class="mark">QB</div>
      <div>
        <div class="bname">QuizBot</div>
        <div class="beyebrow">ESPRIT TUNIS</div>
      </div>
    </div>

    <h1 class="rise d2">Le cours devient<br><span class="grad">espace vectoriel.</span></h1>

    <p class="lede rise d3">
      Chaque support depose est decoupe, vectorise, puis interroge par sens.
      Les passages qui remontent servent a rediger le questionnaire, relu
      question par question avant de vous etre soumis.
    </p>

    <div class="metrics rise d4">
      <div class="metric"><b>384<em>d</em></b><span>DIMENSIONS</span></div>
      <div class="metric"><b>cosinus</b><span>DISTANCE</span></div>
      <div class="metric"><b>local</b><span>INFERENCE</span></div>
    </div>
  </div>
</div>

<script>
(function () {
  var stage = document.getElementById('stage');
  var space = document.getElementById('space');
  var content = document.getElementById('content');
  var canvas = document.getElementById('cloud');
  var ctx = canvas.getContext('2d');
  var calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);

  function resize() {
    var r = stage.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- nuage de points : coordonnees 3D reelles dans un volume centre ------
  var N = 118, FOCAL = 620, pts = [];
  for (var i = 0; i < N; i++) {
    // certains points sont marques "retenus" : ils pulsent en cramoisi, comme
    // les segments que le retrieval fait remonter.
    pts.push({
      x: (Math.random() - 0.5) * 1250,
      y: (Math.random() - 0.5) * 700,
      z: (Math.random() - 0.5) * 900,
      hit: Math.random() < 0.13,
      ph: Math.random() * Math.PI * 2
    });
  }

  function project(p, ry, rx) {
    // rotation autour de Y puis de X, puis projection perspective.
    var cy = Math.cos(ry), sy = Math.sin(ry);
    var x1 = p.x * cy - p.z * sy;
    var z1 = p.x * sy + p.z * cy;
    var cx = Math.cos(rx), sx = Math.sin(rx);
    var y1 = p.y * cx - z1 * sx;
    var z2 = p.y * sx + z1 * cx;
    var s = FOCAL / (FOCAL + z2 + 620);
    return { sx: W / 2 + x1 * s, sy: H * 0.44 + y1 * s, s: s, hit: p.hit, ph: p.ph };
  }

  var tiltX = 0, tiltY = 0, curX = 0, curY = 0, spin = 0, t = 0;

  stage.addEventListener('pointermove', function (e) {
    var r = stage.getBoundingClientRect();
    tiltY = ((e.clientX - r.left) / r.width - 0.5);
    tiltX = ((e.clientY - r.top) / r.height - 0.5);
  });
  stage.addEventListener('pointerleave', function () { tiltX = 0; tiltY = 0; });

  function frame() {
    t += 0.016;
    // inertie : la scene rattrape le pointeur, elle ne s'y colle pas.
    curX += (tiltX - curX) * 0.055;
    curY += (tiltY - curY) * 0.055;
    if (!calm) spin += 0.0013;

    // --- plans CSS 3D : rotation du volume entier ---
    var ry = (-curY * 15).toFixed(2), rx = (curX * 10).toFixed(2);
    space.style.transform = 'rotateY(' + ry + 'deg) rotateX(' + rx + 'deg)';
    // le contenu derive a contre-sens, plus doucement : separation des plans.
    content.style.transform =
      'translate3d(' + (curY * 26).toFixed(2) + 'px,' + (curX * 15).toFixed(2) + 'px,0)';

    // --- nuage ---
    ctx.clearRect(0, 0, W, H);
    var rotY = spin + curY * 0.5, rotX = -curX * 0.32;
    var proj = [];
    for (var i = 0; i < pts.length; i++) proj.push(project(pts[i], rotY, rotX));

    // aretes entre points proches : la structure du voisinage semantique.
    ctx.lineWidth = 1;
    for (var a = 0; a < proj.length; a++) {
      for (var b = a + 1; b < proj.length; b++) {
        var dx = proj[a].sx - proj[b].sx, dy = proj[a].sy - proj[b].sy;
        var d2 = dx * dx + dy * dy;
        if (d2 < 9200) {
          var al = (1 - d2 / 9200) * 0.30 * Math.min(proj[a].s, proj[b].s);
          var lit = proj[a].hit || proj[b].hit;
          ctx.strokeStyle = lit
            ? 'rgba(255,70,100,' + (al * 1.5).toFixed(3) + ')'
            : 'rgba(130,180,255,' + al.toFixed(3) + ')';
          ctx.beginPath();
          ctx.moveTo(proj[a].sx, proj[a].sy);
          ctx.lineTo(proj[b].sx, proj[b].sy);
          ctx.stroke();
        }
      }
    }

    for (var k = 0; k < proj.length; k++) {
      var p = proj[k];
      var pulse = p.hit ? 0.6 + 0.4 * Math.sin(t * 2 + p.ph) : 1;
      var rad = Math.max(0.4, p.s * (p.hit ? 2.5 : 1.5));
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, rad, 0, 6.2832);
      ctx.fillStyle = p.hit
        ? 'rgba(255,72,102,' + (0.92 * pulse * p.s).toFixed(3) + ')'
        : 'rgba(158,200,255,' + (0.62 * p.s).toFixed(3) + ')';
      ctx.fill();
      if (p.hit) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, rad * 3.4 * pulse, 0, 6.2832);
        ctx.fillStyle = 'rgba(255,72,102,' + (0.10 * pulse * p.s).toFixed(3) + ')';
        ctx.fill();
      }
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""
