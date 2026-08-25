"""
Composant de visualisation de la carte sémantique.

Rendu via st.components.v1.html (les <script> sont retirés des blocs
st.markdown) : le nuage est dessiné sur <canvas>, avec survol interactif pour
lire le passage correspondant à chaque point.

Lecture de la carte :
  - deux points proches = deux passages traitant de sujets proches ;
  - un amas = une section conceptuellement homogène du cours ;
  - en mode performance, la couleur donne le score moyen obtenu par les
    étudiants sur les questions issues de ce passage — une zone rouge signale
    une région du cours mal comprise, une zone grise n'a jamais été évaluée.
"""
from __future__ import annotations

import json

MAP_HEIGHT = 540


def build_map_html(payload: dict, show_performance: bool) -> str:
    """Construit le HTML du composant à partir de la réponse de /documents/{id}/map."""
    data = json.dumps(payload.get("points", []))
    method = payload.get("method", "acp")
    perf = "true" if show_performance else "false"

    return r"""
<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:transparent;font-family:'IBM Plex Sans',system-ui,sans-serif}
  #wrap{position:relative;height:520px;border-radius:16px;overflow:hidden;
        background:radial-gradient(120% 120% at 70% -10%,#101C36,#070C18 60%,#04070E);
        border:1px solid rgba(140,180,240,.14)}
  canvas{display:block;width:100%;height:100%;cursor:crosshair}
  #tip{position:absolute;pointer-events:none;max-width:330px;padding:11px 13px;
       border-radius:11px;background:rgba(10,16,32,.97);border:1px solid rgba(140,180,240,.28);
       color:#E8EEF9;font-size:12.5px;line-height:1.5;opacity:0;transition:opacity .13s ease;
       box-shadow:0 20px 46px -18px rgba(0,0,0,.95);z-index:5}
  #tip .m{font-family:'IBM Plex Mono';font-size:9.5px;letter-spacing:.16em;
          text-transform:uppercase;color:#7FB4FF;margin-bottom:6px}
  #tip .s{font-family:'IBM Plex Mono';font-size:11px;margin-top:7px;
          padding-top:7px;border-top:1px solid rgba(140,180,240,.18)}
  #legend{position:absolute;left:14px;bottom:13px;display:flex;gap:15px;flex-wrap:wrap;
          font-family:'IBM Plex Mono';font-size:9.5px;letter-spacing:.13em;
          text-transform:uppercase;color:#8899B8;z-index:4}
  #legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;
            vertical-align:middle}
  #meta{position:absolute;right:14px;top:13px;font-family:'IBM Plex Mono';font-size:9.5px;
        letter-spacing:.14em;text-transform:uppercase;color:#5B6B8A;z-index:4}
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="tip"></div>
  <div id="meta">__METHOD__ · __N__ segments</div>
  <div id="legend"></div>
</div>
<script>
(function(){
  var PTS = __DATA__, PERF = __PERF__;
  var wrap=document.getElementById('wrap'), cv=document.getElementById('c');
  var ctx=cv.getContext('2d'), tip=document.getElementById('tip');
  var dpr=Math.min(window.devicePixelRatio||1,2), W=0,H=0,PAD=46;

  function resize(){
    var r=wrap.getBoundingClientRect(); W=r.width; H=r.height;
    cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); draw();
  }
  function sx(p){ return PAD + p.x*(W-2*PAD); }
  function sy(p){ return PAD + (1-p.y)*(H-2*PAD); }

  function colorOf(p){
    if(!PERF) return {fill:'rgba(127,180,255,.85)', glow:'rgba(127,180,255,.16)'};
    if(p.score===null||p.score===undefined)
      return {fill:'rgba(110,125,155,.5)', glow:'rgba(110,125,155,.08)'};
    // rouge (0) -> ambre (0.5) -> vert (1)
    var s=Math.max(0,Math.min(1,p.score));
    var r = s<.5 ? 255 : Math.round(255-(s-.5)*2*209);
    var g = s<.5 ? Math.round(51+s*2*140) : Math.round(191+(s-.5)*2*39);
    var b = s<.5 ? Math.round(80-s*2*30) : Math.round(50+(s-.5)*2*110);
    return {fill:'rgba('+r+','+g+','+b+',.9)', glow:'rgba('+r+','+g+','+b+',.2)'};
  }
  function radiusOf(p){
    if(!PERF) return 5;
    return p.attempts>0 ? 5+Math.min(p.attempts,6)*0.7 : 3.6;
  }

  var hover=-1;
  function draw(){
    ctx.clearRect(0,0,W,H);

    // grille de reperage
    ctx.strokeStyle='rgba(140,180,240,.055)'; ctx.lineWidth=1;
    for(var i=0;i<=8;i++){
      var gx=PAD+i*(W-2*PAD)/8, gy=PAD+i*(H-2*PAD)/8;
      ctx.beginPath(); ctx.moveTo(gx,PAD); ctx.lineTo(gx,H-PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(W-PAD,gy); ctx.stroke();
    }

    // liens entre points proches : rend la structure des amas lisible
    ctx.lineWidth=1;
    for(var a=0;a<PTS.length;a++){
      for(var b=a+1;b<PTS.length;b++){
        var dx=sx(PTS[a])-sx(PTS[b]), dy=sy(PTS[a])-sy(PTS[b]);
        var d2=dx*dx+dy*dy;
        if(d2<3400){
          ctx.strokeStyle='rgba(127,180,255,'+((1-d2/3400)*0.17).toFixed(3)+')';
          ctx.beginPath(); ctx.moveTo(sx(PTS[a]),sy(PTS[a]));
          ctx.lineTo(sx(PTS[b]),sy(PTS[b])); ctx.stroke();
        }
      }
    }

    for(var k=0;k<PTS.length;k++){
      var p=PTS[k], c=colorOf(p), rad=radiusOf(p), on=(k===hover);
      ctx.beginPath(); ctx.arc(sx(p),sy(p),rad*(on?3.4:2.6),0,6.2832);
      ctx.fillStyle=c.glow; ctx.fill();
      ctx.beginPath(); ctx.arc(sx(p),sy(p),on?rad*1.45:rad,0,6.2832);
      ctx.fillStyle=c.fill; ctx.fill();
      if(on){ ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke(); }
    }
  }

  cv.addEventListener('mousemove',function(e){
    var r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    var best=-1,bd=1e9;
    for(var k=0;k<PTS.length;k++){
      var dx=sx(PTS[k])-mx, dy=sy(PTS[k])-my, d=dx*dx+dy*dy;
      if(d<bd){bd=d;best=k;}
    }
    if(bd<420){
      if(best!==hover){hover=best;draw();}
      var p=PTS[best];
      var score = !PERF ? ''
        : (p.score===null||p.score===undefined
            ? '<div class="s">jamais évalué par un quiz</div>'
            : '<div class="s">score moyen '+(p.score*100).toFixed(0)+'% · '+p.attempts+' réponse(s)</div>');
      tip.innerHTML='<div class="m">segment '+p.chunk_index+
                    (p.page?' · page '+p.page:'')+'</div>'+
                    p.preview.replace(/&/g,'&amp;').replace(/</g,'&lt;')+score;
      var tw=Math.min(330,W*0.6);
      tip.style.left=Math.min(mx+16,W-tw-12)+'px';
      tip.style.top=Math.max(8,my-14)+'px';
      tip.style.opacity=1;
    } else {
      if(hover!==-1){hover=-1;draw();}
      tip.style.opacity=0;
    }
  });
  cv.addEventListener('mouseleave',function(){hover=-1;tip.style.opacity=0;draw();});

  var lg=document.getElementById('legend');
  lg.innerHTML = PERF
    ? '<span><i style="background:rgba(255,51,80,.9)"></i>mal compris</span>'+
      '<span><i style="background:rgba(255,191,50,.9)"></i>partiel</span>'+
      '<span><i style="background:rgba(46,230,160,.9)"></i>maîtrisé</span>'+
      '<span><i style="background:rgba(110,125,155,.6)"></i>jamais évalué</span>'
    : '<span><i style="background:rgba(127,180,255,.85)"></i>segment de cours · proximité = proximité de sens</span>';

  window.addEventListener('resize',resize);
  resize();
})();
</script></body></html>
""".replace("__DATA__", data).replace("__PERF__", perf) \
   .replace("__METHOD__", method).replace("__N__", str(payload.get("num_points", 0)))
