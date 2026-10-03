"""Read-only localhost dashboard for single- or three-server PPO V2 logs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from config.scenario_labels import training_display_scenario_id


TOWNS = ("Town02", "Town05", "Town10HD")
SCENARIOS = ("s1", "s2", "s3", "s4", "s5", "s6")
TOWN_SCENARIOS = {
    "Town02": ("s1", "s6"),
    "Town05": ("s2", "s5"),
    "Town10HD": ("s3", "s4"),
}


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__EXPERIMENT_NAME__</title>
<style>
:root{color-scheme:dark;--bg:#080d1b;--card:#111a2e;--line:#27324a;--muted:#94a3b8;--text:#e8edf7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;gap:20px;align-items:center;flex-wrap:wrap}
h1{font-size:19px;margin:0}.muted{color:var(--muted)}.ok{color:#4ade80}.wait{color:#fbbf24}.bad{color:#fb7185}
.assessment{margin:12px 12px 0;padding:11px 14px;background:var(--card);border:1px solid var(--line);border-radius:10px}
.section-title{margin:18px 14px 8px;font-size:15px;font-weight:700;color:#cbd5e1}
.town-grid,.scenario-grid,.validation-grid,.global-grid,.experiment-grid{display:grid;gap:12px;padding:0 12px}
.town-grid,.experiment-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.scenario-grid,.validation-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.global-grid{grid-template-columns:repeat(2,minmax(0,1fr));padding-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px;min-width:0}
.title-row{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}.title{font-weight:650}.meta{font-size:12px;color:var(--muted);white-space:nowrap}
.progress{height:5px;background:#1e293b;border-radius:5px;overflow:hidden;margin:0 0 5px}.progress>span{display:block;height:100%;background:#38bdf8;width:0}
.breakdown{font-size:12px;color:var(--muted);margin-top:8px;min-height:18px}.town-card{padding:14px}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
canvas{width:100%;height:205px;display:block}.global-grid canvas{height:285px}
@media(max-width:1050px){.town-grid,.scenario-grid,.validation-grid,.experiment-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:720px){.town-grid,.scenario-grid,.validation-grid,.global-grid,.experiment-grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>__EXPERIMENT_NAME__</h1><span id="status" class="wait">connecting</span><span id="counts" class="muted"></span><span id="source" class="muted"></span></header>
<section class="assessment"><strong id="assessment-state">Assessing…</strong><span id="assessment-text" class="muted"></span></section>

<div class="section-title">Experiment — total progress and ETA</div>
<main class="experiment-grid">
  <section class="card town-card"><div class="title-row"><span class="title">Overall pipeline</span><span id="experiment-meta" class="meta"></span></div><div class="progress"><span id="experiment-progress"></span></div><div id="experiment-stage" class="breakdown"></div></section>
  <section class="card town-card"><div class="title-row"><span class="title">Training transitions</span><span id="training-meta" class="meta"></span></div><div class="progress"><span id="training-progress"></span></div><div id="training-stage" class="breakdown"></div></section>
  <section class="card town-card"><div class="title-row"><span class="title">Time estimate</span><span id="eta-meta" class="meta"></span></div><div id="eta-detail" class="breakdown">Waiting for timing samples</div></section>
</main>

<div class="section-title">Three-server rollout — live Town progress</div>
<main class="town-grid">
  <section class="card town-card"><div class="title-row"><span class="title">Town02 · S5/S6</span><span id="town02-meta" class="meta"></span></div><div class="progress"><span id="town02-progress"></span></div><div id="town02-breakdown" class="breakdown"></div></section>
  <section class="card town-card"><div class="title-row"><span class="title">Town05 · S2/S1</span><span id="town05-meta" class="meta"></span></div><div class="progress"><span id="town05-progress"></span></div><div id="town05-breakdown" class="breakdown"></div></section>
  <section class="card town-card"><div class="title-row"><span class="title">Town10HD · S3/S4</span><span id="town10hd-meta" class="meta"></span></div><div class="progress"><span id="town10hd-progress"></span></div><div id="town10hd-breakdown" class="breakdown"></div></section>
</main>

<div class="section-title">Scenario reward — six independent views</div>
<main class="scenario-grid">
  <section class="card"><div class="title-row"><span class="title">S5 · High Traffic</span><span id="s5-meta" class="meta"></span></div><canvas id="s5"></canvas></section>
  <section class="card"><div class="title-row"><span class="title">S2 · Curved</span><span id="s2-meta" class="meta"></span></div><canvas id="s2"></canvas></section>
  <section class="card"><div class="title-row"><span class="title">S3 · Corridor</span><span id="s3-meta" class="meta"></span></div><canvas id="s3"></canvas></section>
  <section class="card"><div class="title-row"><span class="title">S6 · Lane Closure</span><span id="s6-meta" class="meta"></span></div><canvas id="s6"></canvas></section>
  <section class="card"><div class="title-row"><span class="title">S1 · Normal</span><span id="s1-meta" class="meta"></span></div><canvas id="s1"></canvas></section>
  <section class="card"><div class="title-row"><span class="title">S4 · Jaywalker</span><span id="s4-meta" class="meta"></span></div><canvas id="s4"></canvas></section>
</main>

<div class="section-title">Global summary and PPO optimization</div>
<main class="global-grid">
  <section class="card"><div class="title">All-scenario reward vs environment steps</div><canvas id="reward"></canvas></section>
  <section class="card"><div class="title">Rolling success rate vs environment steps</div><canvas id="success"></canvas></section>
  <section class="card"><div class="title">Policy / value loss vs environment steps</div><canvas id="loss"></canvas></section>
  <section class="card"><div class="title">Approx KL monitor / PPO clip fraction vs environment steps</div><canvas id="stability"></canvas></section>
</main>

<div class="section-title">Held-out validation — 180 episodes per policy</div>
<main class="global-grid">
  <section class="card"><div class="title">Validation status</div><div id="validation-latest" class="breakdown">Waiting for first completed validation</div></section>
  <section class="card"><div class="title">Macro validation rates vs policy update</div><canvas id="validation"></canvas></section>
</main>

<div class="section-title">Held-out validation — six scenario details</div>
<main class="validation-grid">
  <section class="card"><div class="title-row"><span class="title">S5 · High Traffic · Town02</span><span id="val-s5-meta" class="meta">waiting</span></div><div id="val-s5-detail" class="breakdown">Waiting for validation</div><canvas id="val-s5"></canvas><div id="val-s5-reasons" class="breakdown"></div></section>
  <section class="card"><div class="title-row"><span class="title">S2 · Curved · Town05</span><span id="val-s2-meta" class="meta">waiting</span></div><div id="val-s2-detail" class="breakdown">Waiting for validation</div><canvas id="val-s2"></canvas><div id="val-s2-reasons" class="breakdown"></div></section>
  <section class="card"><div class="title-row"><span class="title">S3 · Corridor · Town10HD</span><span id="val-s3-meta" class="meta">waiting</span></div><div id="val-s3-detail" class="breakdown">Waiting for validation</div><canvas id="val-s3"></canvas><div id="val-s3-reasons" class="breakdown"></div></section>
  <section class="card"><div class="title-row"><span class="title">S6 · Lane Closure · Town02</span><span id="val-s6-meta" class="meta">waiting</span></div><div id="val-s6-detail" class="breakdown">Waiting for validation</div><canvas id="val-s6"></canvas><div id="val-s6-reasons" class="breakdown"></div></section>
  <section class="card"><div class="title-row"><span class="title">S1 · Normal · Town05</span><span id="val-s1-meta" class="meta">waiting</span></div><div id="val-s1-detail" class="breakdown">Waiting for validation</div><canvas id="val-s1"></canvas><div id="val-s1-reasons" class="breakdown"></div></section>
  <section class="card"><div class="title-row"><span class="title">S4 · Jaywalker · Town10HD</span><span id="val-s4-meta" class="meta">waiting</span></div><div id="val-s4-detail" class="breakdown">Waiting for validation</div><canvas id="val-s4"></canvas><div id="val-s4-reasons" class="breakdown"></div></section>
</main>
<script>
const palette=['#38bdf8','#fb923c','#c084fc','#f43f5e','#4ade80','#facc15'];
const rolling=(a,n)=>a.map((_,i)=>{let s=0,c=0;for(let j=Math.max(0,i-n+1);j<=i;j++){if(Number.isFinite(a[j])){s+=a[j];c++}}return c?s/c:NaN});
function draw(id,series,empty){
 const c=document.getElementById(id),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
 const g=c.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
 const pts=series.flatMap(s=>s.x.map((v,i)=>[v,s.y[i]])).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
 if(!pts.length){g.fillStyle='#94a3b8';g.textAlign='center';g.fillText(empty,w/2,h/2);return}
 let xmin=Math.min(...pts.map(p=>p[0])),xmax=Math.max(...pts.map(p=>p[0])),ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1]));
 if(/^(?:s|val-s)[1-6]$/.test(id)&&xmin>0)xmin=0;
 if(xmin===xmax){xmin-=1;xmax+=1}if(ymin===ymax){ymin-=1;ymax+=1}const py=Math.max((ymax-ymin)*.08,1e-9);ymin-=py;ymax+=py;
 const L=55,R=12,T=series.length>3?34:17,B=31,X=v=>L+(v-xmin)/(xmax-xmin)*(w-L-R),Y=v=>T+(ymax-v)/(ymax-ymin)*(h-T-B);
 g.strokeStyle='#334155';g.fillStyle='#94a3b8';g.lineWidth=1;g.font='10px system-ui';g.textAlign='right';
 for(let i=0;i<=3;i++){let yy=T+i*(h-T-B)/3,val=ymax-i*(ymax-ymin)/3;g.beginPath();g.moveTo(L,yy);g.lineTo(w-R,yy);g.stroke();g.fillText(val.toPrecision(4),L-6,yy+3)}
 g.textAlign='center';for(let i=0;i<=3;i++){let xx=L+i*(w-L-R)/3,val=xmin+i*(xmax-xmin)/3;g.fillText(Math.round(val).toLocaleString(),xx,h-9)}
 series.forEach((s,k)=>{g.strokeStyle=s.color||palette[k%palette.length];g.globalAlpha=s.alpha??1;g.lineWidth=s.width||2;g.beginPath();let on=false;s.x.forEach((v,i)=>{let y=s.y[i];if(!Number.isFinite(v)||!Number.isFinite(y)){on=false;return}on?g.lineTo(X(v),Y(y)):g.moveTo(X(v),Y(y));on=true});g.stroke();g.globalAlpha=1;g.fillStyle=s.color||palette[k%palette.length];g.textAlign='left';const columns=series.length>3?3:series.length;g.fillText(s.name,L+7+(k%columns)*(w-L-R)/columns,(series.length>3?11:T+10)+Math.floor(k/columns)*13)});
}
function rewardSeries(rows,color){const x=rows.map(v=>Number.isFinite(v.scenario_step)?v.scenario_step:v.global_step),y=rows.map(v=>v.episode_return);return[{name:'episode',x,y,color,alpha:.35,width:1},{name:'mean(20)',x,y:rolling(y,20),color,width:3}]}
function duration(seconds){if(!Number.isFinite(seconds))return 'estimating';seconds=Math.max(0,Math.round(seconds));const d=Math.floor(seconds/86400),h=Math.floor(seconds%86400/3600),m=Math.floor(seconds%3600/60);return `${d?d+'d ':''}${h?String(h).padStart(2,'0')+':':''}${String(m).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`}
async function refresh(){try{const response=await fetch('/api/data',{cache:'no-store'}),d=await response.json();
 document.getElementById('status').textContent='live';document.getElementById('status').className='ok';
 document.getElementById('counts').textContent=`episodes ${d.episodes.length} · updates ${d.assessment.update_count} · ${d.run_status}`;
 document.getElementById('source').textContent=d.episode_sources.join(' + ');
 const a=d.assessment;document.getElementById('assessment-state').textContent=`${a.label} · `;document.getElementById('assessment-state').className=a.level;document.getElementById('assessment-text').textContent=a.message;
 const e=d.experiment||{},total=e.total_updates||0,overall=100*(e.overall_fraction||0),training=100*(e.training_fraction||0),stage=String(e.stage||'waiting').replaceAll('_',' ');
 document.getElementById('experiment-meta').textContent=`${overall.toFixed(1)}%`;
 document.getElementById('experiment-progress').style.width=`${Math.min(100,overall)}%`;
 document.getElementById('experiment-stage').textContent=`Update ${e.current_update||0}/${total} · ${stage} · trained ${e.completed_training_updates||0}/${total} · validated ${e.completed_validation_updates||0}/${total}${e.stage==='validation'?` · scenarios ${e.validation_scenarios_finished}/6`:''}`;
 document.getElementById('training-meta').textContent=`${training.toFixed(1)}%`;
 document.getElementById('training-progress').style.width=`${Math.min(100,training)}%`;
 document.getElementById('training-stage').textContent=`${Number(e.training_steps||0).toLocaleString()} / ${Number(e.total_training_steps||0).toLocaleString()} transitions`;
 document.getElementById('eta-meta').textContent=Number.isFinite(e.eta_seconds)?`ETA ${duration(e.eta_seconds)}`:'estimating';
 document.getElementById('eta-detail').textContent=`Elapsed ${duration(e.elapsed_seconds)}${Number.isFinite(e.seconds_per_update)?` · avg/update ${duration(e.seconds_per_update)} · finish ${e.estimated_finish||'—'} · samples ${e.eta_samples}`:' · waiting for enough progress'}`;
 d.workers.forEach(worker=>{const key=worker.town.toLowerCase(),pct=100*worker.current_step/Math.max(worker.target_steps,1);document.getElementById(`${key}-meta`).textContent=`${worker.current_step.toLocaleString()}/${worker.target_steps.toLocaleString()} · ${pct.toFixed(1)}% · ep ${worker.current_episodes}`;document.getElementById(`${key}-progress`).style.width=`${Math.min(100,pct)}%`;const parts=Object.entries(worker.scenario_steps).map(([name,steps])=>`${name.toUpperCase()} ${steps.toLocaleString()}`);if(worker.in_flight_steps>0)parts.push(`in-flight ${worker.in_flight_steps.toLocaleString()}`);document.getElementById(`${key}-breakdown`).textContent=parts.join(' · ')});
 ['s1','s2','s3','s4','s5','s6'].forEach((scenario,index)=>{const rows=d.scenarios[scenario]||[],progress=d.scenario_progress[scenario];document.getElementById(`${scenario}-meta`).textContent=`${progress.exact?'steps':'completed steps'} ${progress.steps.toLocaleString()} · episodes ${rows.length}`;draw(scenario,rewardSeries(rows,palette[index]),'No completed episodes yet')});
 const ex=d.episodes.map(v=>v.global_step),rew=d.episodes.map(v=>v.episode_return),suc=d.episodes.map(v=>100*v.success),ux=d.epochs.map(v=>v.environment_steps);
 draw('reward',[{name:'episode',x:ex,y:rew,color:'#64748b',alpha:.45,width:1},{name:'mean(20)',x:ex,y:rolling(rew,20),color:'#38bdf8',width:3}],'Waiting for completed episodes');
 draw('success',[{name:'success %(50)',x:ex,y:rolling(suc,50),color:'#4ade80',width:3}],'Waiting for completed episodes');
 draw('loss',[{name:'policy',x:ux,y:d.epochs.map(v=>v.policy_loss),color:'#38bdf8'},{name:'value',x:ux,y:d.epochs.map(v=>v.value_loss),color:'#fb923c'}],'Waiting for first PPO update');
 draw('stability',[{name:'approx KL',x:ux,y:d.epochs.map(v=>v.approx_kl),color:'#c084fc'},{name:'clip fraction',x:ux,y:d.epochs.map(v=>v.clip_fraction),color:'#f43f5e'}],'Waiting for first PPO update');
 const vr=d.validation||[],vx=vr.map(v=>v.update),last=vr.length?vr[vr.length-1]:null;
 if(last){const rows=Object.entries(last.scenario_metrics||{}).map(([scenario,m])=>`<tr><td>${scenario.toUpperCase()}</td><td>${(100*m.success_rate).toFixed(1)}%</td><td>${(100*m.collision_rate).toFixed(1)}%</td><td>${(100*m.workzone_violation_rate).toFixed(1)}%</td><td>${(100*m.off_road_rate).toFixed(1)}%</td><td>${(100*m.timeout_rate).toFixed(1)}%</td><td>${Number(m.mean_episode_return).toFixed(1)}</td></tr>`).join('');document.getElementById('validation-latest').innerHTML=`<div>Policy ${last.update} · ${last.episodes} episodes · success ${(100*last.success_rate).toFixed(1)}% · collision ${(100*last.collision_rate).toFixed(1)}% · violation ${(100*last.violation_rate).toFixed(1)}% · off-road ${(100*last.off_road_rate).toFixed(1)}% · timeout ${(100*last.timeout_rate).toFixed(1)}% · mean return ${last.mean_return.toFixed(1)} · best policy ${last.best_update}${last.best_updated?' (new best)':''}</div>${rows?`<table><thead><tr><th>Scenario</th><th>Success</th><th>Collision</th><th>Violation</th><th>Off-road</th><th>Timeout</th><th>Return</th></tr></thead><tbody>${rows}</tbody></table>`:''}`}else{document.getElementById('validation-latest').textContent='Waiting for first completed validation'};
 draw('validation',[{name:'success %',x:vx,y:vr.map(v=>100*v.success_rate),color:'#4ade80'},{name:'safety failure %',x:vx,y:vr.map(v=>100*v.safety_failure_rate),color:'#fb7185'},{name:'timeout %',x:vx,y:vr.map(v=>100*v.timeout_rate),color:'#fbbf24'}],'Waiting for first completed validation');
 ['s1','s2','s3','s4','s5','s6'].forEach((scenario,index)=>{
  const rows=vr.map(v=>({update:v.update,metrics:(v.scenario_metrics||{})[scenario]})).filter(v=>v.metrics),latest=rows.length?rows[rows.length-1]:null;
  if(latest){const m=latest.metrics,reasons=Object.entries(m.reason_counts||{}).sort((a,b)=>b[1]-a[1]).map(([reason,count])=>`${reason.replaceAll('_',' ')} ${count}`).join(' · ');document.getElementById(`val-${scenario}-meta`).textContent=`U${latest.update} · ${m.episodes||0} episodes`;document.getElementById(`val-${scenario}-detail`).textContent=`success ${(100*m.success_rate).toFixed(1)}% · collision ${(100*m.collision_rate).toFixed(1)}% · work-zone ${(100*m.workzone_violation_rate).toFixed(1)}% · off-road ${(100*m.off_road_rate).toFixed(1)}% · timeout ${(100*m.timeout_rate).toFixed(1)}% · return ${Number(m.mean_episode_return).toFixed(1)}`;document.getElementById(`val-${scenario}-reasons`).textContent=reasons?`end reasons · ${reasons}`:'end reasons unavailable'}else{document.getElementById(`val-${scenario}-meta`).textContent='waiting';document.getElementById(`val-${scenario}-detail`).textContent='Waiting for validation';document.getElementById(`val-${scenario}-reasons`).textContent=''}
  const x=rows.map(v=>v.update),metrics=rows.map(v=>v.metrics);draw(`val-${scenario}`,[{name:'success %',x,y:metrics.map(m=>100*m.success_rate),color:'#4ade80'},{name:'collision %',x,y:metrics.map(m=>100*m.collision_rate),color:'#fb7185'},{name:'work-zone %',x,y:metrics.map(m=>100*m.workzone_violation_rate),color:'#c084fc'},{name:'off-road %',x,y:metrics.map(m=>100*m.off_road_rate),color:'#38bdf8'},{name:'timeout %',x,y:metrics.map(m=>100*m.timeout_rate),color:'#fbbf24'}],'Waiting for validation');
 });
 }catch(error){document.getElementById('status').textContent='waiting for telemetry';document.getElementById('status').className='wait'}}
refresh();setInterval(refresh,2000);addEventListener('resize',refresh);
</script></body></html>"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        return []
    return records


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _timestamp_seconds(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OSError):
        return None


def _validation_scenarios_finished(run_root: Path, update: int) -> int:
    if update < 1:
        return 0
    update_dir = run_root / "validation" / f"update_{update:03d}"
    return sum(
        1
        for scenario in SCENARIOS
        if any((update_dir / scenario).glob("summary_*.json"))
    )


def _experiment_progress(
    *,
    log_dir: Path,
    state: dict[str, Any],
    workers: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    raw_episodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    total_updates: int | None,
    configured_steps_per_update: int | None,
) -> dict[str, Any]:
    plan = state.get("plan") or {}
    configured_total = int(total_updates or plan.get("total_updates") or 0)
    steps_per_update = int(
        configured_steps_per_update or plan.get("steps_per_update") or 0
    )
    completed_training = max(
        [int(item.get("update", 0) or 0) for item in state.get("completed_updates", [])]
        + [max(0, int(state.get("next_update", 1) or 1) - 1)],
        default=0,
    )
    validated_updates = sorted({int(item["update"]) for item in validation})
    completed_validation = len(validated_updates)
    active = state.get("active_update") or {}
    active_update = int(active.get("index", 0) or 0)
    active_state = str(active.get("state", ""))

    current_update = active_update
    cycle_fraction = 0.0
    validation_finished = 0
    if configured_total > 0 and completed_validation >= configured_total:
        stage = "complete"
        current_update = configured_total
    elif active_update:
        if active_state == "collecting":
            stage = "rollout"
            target = sum(max(0, int(worker.get("target_steps", 0))) for worker in workers)
            current = sum(max(0, int(worker.get("current_step", 0))) for worker in workers)
            cycle_fraction = 0.75 * min(1.0, current / max(1, target))
        else:
            stage = "ppo"
            latest_environment_step = max(
                (_number(row, "environment_steps") or 0.0 for row in epochs),
                default=0.0,
            )
            current_epochs = [
                row for row in epochs
                if (_number(row, "environment_steps") or -1.0) == latest_environment_step
            ]
            epoch = max((_number(row, "epoch") or 0.0 for row in current_epochs), default=0.0)
            epochs_total = max(
                (_number(row, "epochs_total") or 0.0 for row in current_epochs),
                default=0.0,
            )
            cycle_fraction = 0.75 + 0.10 * min(1.0, epoch / max(1.0, epochs_total))
    elif completed_training > completed_validation:
        stage = "validation"
        current_update = completed_training
        validation_finished = _validation_scenarios_finished(log_dir.parent, current_update)
        cycle_fraction = 0.85 + 0.15 * validation_finished / len(SCENARIOS)
    elif configured_total > 0 and completed_training >= configured_total:
        stage = "complete"
        current_update = configured_total
    elif state:
        stage = "between_updates"
        current_update = min(configured_total or completed_training + 1, completed_training + 1)
    else:
        stage = "waiting"
        current_update = 1 if configured_total else 0

    current_rollout_steps = 0
    if stage in {"rollout", "ppo"}:
        current_rollout_steps = sum(
            max(0, int(worker.get("current_step", 0))) for worker in workers
        )
    training_steps = completed_training * steps_per_update
    if active_update:
        training_steps = max(
            0,
            (active_update - 1) * steps_per_update + min(steps_per_update, current_rollout_steps),
        )
    total_training_steps = configured_total * steps_per_update
    training_fraction = (
        min(1.0, training_steps / total_training_steps)
        if total_training_steps > 0 else 0.0
    )
    equivalent_cycles = min(
        float(configured_total or completed_validation + 1),
        completed_validation + cycle_fraction,
    )
    overall_fraction = (
        equivalent_cycles / configured_total if configured_total > 0 else 0.0
    )

    timestamp_candidates = [
        _timestamp_seconds(row.get("timestamp"))
        for row in [*raw_episodes, *epochs, *events]
    ]
    timestamps = [value for value in timestamp_candidates if value is not None]
    started_at = min(timestamps) if timestamps else None
    validation_times = sorted(
        value
        for event in events
        if event.get("event") == "validation_complete"
        if (value := _timestamp_seconds(event.get("timestamp"))) is not None
    )
    cycle_samples: list[float] = []
    if len(validation_times) >= 2:
        recent_validation_times = validation_times[-6:]
        cycle_samples = [
            later - earlier
            for earlier, later in zip(
                recent_validation_times,
                recent_validation_times[1:],
            )
            if later > earlier
        ]
    if not cycle_samples and validation_times and started_at is not None:
        first_duration = validation_times[0] - started_at
        if first_duration > 0:
            cycle_samples = [first_duration]
    now = time.time()
    if not cycle_samples and started_at is not None and equivalent_cycles > 0.02:
        cycle_samples = [(now - started_at) / equivalent_cycles]
    seconds_per_update = (
        sum(cycle_samples) / len(cycle_samples) if cycle_samples else None
    )
    eta_seconds = None
    estimated_finish = None
    if seconds_per_update is not None and configured_total > 0:
        eta_seconds = max(0.0, seconds_per_update * (configured_total - equivalent_cycles))
        estimated_finish = datetime.fromtimestamp(now + eta_seconds).astimezone().isoformat(
            timespec="minutes"
        )

    return {
        "total_updates": configured_total,
        "current_update": current_update,
        "completed_training_updates": completed_training,
        "completed_validation_updates": completed_validation,
        "stage": stage,
        "validation_scenarios_finished": validation_finished,
        "training_steps": training_steps,
        "total_training_steps": total_training_steps,
        "training_fraction": training_fraction,
        "overall_fraction": max(0.0, min(1.0, overall_fraction)),
        "elapsed_seconds": max(0.0, now - started_at) if started_at is not None else None,
        "eta_seconds": eta_seconds,
        "seconds_per_update": seconds_per_update,
        "estimated_finish": estimated_finish,
        "eta_samples": len(cycle_samples),
    }


def _latest_run(rows: list[dict[str, Any]], step_key: str) -> list[dict[str, Any]]:
    start = 0
    previous: float | None = None
    for index, row in enumerate(rows):
        current = _number(row, step_key)
        if current is None:
            continue
        if previous is not None and current < previous:
            start = index
        previous = current
    return rows[start:]


def _latest_csv(log_dir: Path, prefix: str) -> Path:
    timestamped = sorted(
        log_dir.glob(f"{prefix}_*.csv"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    return timestamped[-1] if timestamped else log_dir / f"{prefix}.csv"


def _latest_telemetry_paths(log_dir: Path) -> tuple[Path, Path]:
    episodes_path = _latest_csv(log_dir, "episodes")
    if episodes_path.name == "episodes.csv":
        return episodes_path, log_dir / "ppo_epochs.csv"
    stamp = episodes_path.stem[len("episodes_"):]
    return episodes_path, log_dir / f"ppo_epochs_{stamp}.csv"


def _worker_episode_paths(log_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for town in TOWNS:
        directory = log_dir / "workers" / town
        path = _latest_csv(directory, "episodes")
        if path.is_file():
            result[town] = path
    return result


def _read_state(log_dir: Path) -> dict[str, Any]:
    path = log_dir.parent / "run_state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}


def _episode_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("global_step"),
        row.get("town"),
        row.get("setting_id"),
        row.get("origin_index"),
        row.get("episode_length"),
        row.get("reason"),
    )


def _deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        unique[_episode_identity(row)] = row
    return sorted(
        unique.values(),
        key=lambda row: (_number(row, "global_step") or -1.0, str(row.get("timestamp", ""))),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _assessment(
    episodes: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
) -> dict[str, Any]:
    update_steps = sorted({
        value for row in epochs
        if (value := _number(row, "environment_steps")) is not None
    })
    update_count = len(update_steps)
    rewards = [
        value for row in episodes
        if (value := _number(row, "episode_return")) is not None
    ]
    successes = [
        value for row in episodes
        if (value := _number(row, "success")) is not None
    ]
    latest_step = update_steps[-1] if update_steps else None
    latest_epochs = [row for row in epochs if _number(row, "environment_steps") == latest_step]
    latest_kl = _mean([
        value for row in latest_epochs
        if (value := _number(row, "approx_kl")) is not None
    ])
    latest_clip = _mean([
        value for row in latest_epochs
        if (value := _number(row, "clip_fraction")) is not None
    ])
    base = {"update_count": update_count, "latest_kl": latest_kl, "latest_clip_fraction": latest_clip}
    if update_count == 0:
        return {
            **base, "label": "COLLECTING", "level": "wait",
            "message": f"Three workers are live with {len(rewards)} completed episodes; PPO waits for the all-Town barrier.",
        }
    if latest_kl is not None and latest_clip is not None and (latest_kl > 0.03 or latest_clip > 0.35):
        return {
            **base, "label": "UPDATE UNSTABLE", "level": "bad",
            "message": f"Mean KL={latest_kl:.4f}, clip={latest_clip:.1%}. Reduce PPO epochs or learning rate.",
        }
    if update_count < 5 or len(rewards) < 200 or len(successes) < 200:
        return {
            **base, "label": "WARMING UP", "level": "wait",
            "message": f"Need 5 updates and 200 episodes; currently {update_count} updates and {len(rewards)} episodes.",
        }
    window = 100
    reward_previous = _mean(rewards[-2 * window:-window])
    reward_recent = _mean(rewards[-window:])
    success_previous = _mean(successes[-2 * window:-window])
    success_recent = _mean(successes[-window:])
    assert reward_previous is not None and reward_recent is not None
    assert success_previous is not None and success_recent is not None
    reward_delta = reward_recent - reward_previous
    success_delta = success_recent - success_previous
    base.update({"reward_delta": reward_delta, "success_delta": success_delta})
    if abs(reward_delta) < 2.0 and abs(success_delta) < 0.02:
        return {
            **base, "label": "PLATEAU CANDIDATE", "level": "ok",
            "message": f"Last 100 vs previous 100: reward {reward_delta:+.2f}, success {success_delta:+.1%}.",
        }
    if reward_delta > 0.0 or success_delta > 0.0:
        return {
            **base, "label": "IMPROVING", "level": "ok",
            "message": f"Last 100 vs previous 100: reward {reward_delta:+.2f}, success {success_delta:+.1%}.",
        }
    return {
        **base, "label": "DEGRADING", "level": "bad",
        "message": f"Last 100 vs previous 100: reward {reward_delta:+.2f}, success {success_delta:+.1%}.",
    }


def _public_episode(row: dict[str, Any]) -> dict[str, Any]:
    setting = str(row.get("setting_id", "unknown/unknown/unknown"))
    scenario = str(row.get("scenario") or setting.split("/", 1)[0])
    return {
        "global_step": _number(row, "global_step"),
        "town_step": _number(row, "town_step"),
        "episode_length": int(_number(row, "episode_length") or 0),
        "episode_return": _number(row, "episode_return"),
        "success": _number(row, "success"),
        "town": str(row.get("town", "unknown")),
        "scenario": training_display_scenario_id(scenario),
    }


def _scenario_series(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Give every scenario its own cumulative-step axis."""
    totals = {scenario: 0 for scenario in SCENARIOS}
    grouped: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for episode in episodes:
        scenario = str(episode.get("scenario", "")).lower()
        if scenario not in grouped:
            continue
        totals[scenario] += max(0, int(episode.get("episode_length", 0) or 0))
        point = dict(episode)
        point["scenario_step"] = totals[scenario]
        grouped[scenario].append(point)
    return grouped, totals


def _fragment_scenario_steps(
    path: Path,
    allowed_scenarios: tuple[str, ...],
) -> tuple[int, dict[str, int]] | None:
    """Read exact per-scenario valid steps from a completed Town fragment."""
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        valid_steps = int(descriptor["valid_steps"])
        context_rows = descriptor["context_summary"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(context_rows, list) or valid_steps < 0:
        return None
    steps = {scenario: 0 for scenario in allowed_scenarios}
    try:
        for row in context_rows:
            scenario = str(row.get("scenario", "")).lower()
            if scenario in steps:
                steps[scenario] += max(0, int(row.get("steps", 0)))
    except (AttributeError, TypeError, ValueError):
        return None
    if sum(steps.values()) != valid_steps:
        return None
    return valid_steps, steps


def _plan_context(state: dict[str, Any]) -> tuple[int, int, dict[str, int]]:
    plan = state.get("plan", {})
    steps_per_update = int(plan.get("steps_per_update", 0) or 0)
    active = state.get("active_update") or {}
    update_index = int(active.get("index") or max(1, int(state.get("next_update", 1)) - 1))
    quotas: dict[str, int] = {}
    for phase in plan.get("rollout_and_ppo", {}).get("phases", []):
        quotas[str(phase.get("town"))] = int(phase.get("quota_steps", 0) or 0)
    return update_index, steps_per_update, quotas


def _payload(
    log_dir: Path,
    total_updates: int | None = None,
    steps_per_update: int | None = None,
) -> dict[str, Any]:
    root_episodes_path, epochs_path = _latest_telemetry_paths(log_dir)
    worker_paths = _worker_episode_paths(log_dir)
    if worker_paths:
        # Worker CSVs are authoritative for a three-server run.  The parent
        # later imports the same rows after the barrier, so including both
        # sources would double every completed episode.
        raw_by_town = {
            town: _latest_run(_rows(path), "global_step")
            for town, path in worker_paths.items()
        }
        raw_episodes = _deduplicate(
            row for rows in raw_by_town.values() for row in rows
        )
        episode_sources = [f"{town}:{path.name}" for town, path in worker_paths.items()]
    else:
        raw_episodes = _latest_run(_rows(root_episodes_path), "global_step")
        raw_by_town = {}
        episode_sources = [root_episodes_path.name]
    epochs = _latest_run(_rows(epochs_path), "environment_steps")
    public_episodes = [_public_episode(row) for row in raw_episodes]
    state = _read_state(log_dir)
    events = _jsonl(log_dir / "events.jsonl")
    validation = []
    for event in events:
        if event.get("event") != "validation_complete":
            continue
        metrics = event.get("metrics") or {}
        best = event.get("best_model") or {}
        try:
            validation.append({
                "update": int(event["update"]),
                "success_rate": float(metrics["macro_success_rate"]),
                "collision_rate": float(metrics["macro_collision_rate"]),
                "violation_rate": float(metrics["macro_workzone_violation_rate"]),
                "off_road_rate": float(metrics["macro_off_road_rate"]),
                "timeout_rate": float(metrics["macro_timeout_rate"]),
                "safety_failure_rate": float(metrics["macro_safety_failure_rate"]),
                "mean_return": float(metrics["macro_mean_episode_return"]),
                "episodes": int(metrics.get("episodes", 0)),
                "scenario_metrics": dict(metrics.get("scenario_metrics") or {}),
                "best_updated": bool(event.get("best_updated")),
                "best_update": int(best.get("update", event["update"])),
            })
        except (KeyError, TypeError, ValueError):
            continue
    validation.sort(key=lambda item: item["update"])
    update_index, plan_steps_per_update, quotas = _plan_context(state)
    update_base = max(0, update_index - 1) * plan_steps_per_update
    offset = update_base
    workers = []
    scenario_progress = {
        scenario: {"steps": 0, "exact": False}
        for scenario in SCENARIOS
    }
    for town in TOWNS:
        town_rows = raw_by_town.get(
            town,
            [row for row in raw_episodes if str(row.get("town")) == town],
        )
        target = int(quotas.get(town, 0))
        current_rows = [
            row for row in town_rows
            if (_number(row, "global_step") or -1) >= offset
            and (target <= 0 or (_number(row, "global_step") or -1) < offset + target + 1)
        ]
        public_town = [_public_episode(row) for row in town_rows]
        current_step = max(
            (int(_number(row, "town_step") or 0) for row in current_rows),
            default=0,
        )
        allowed_scenarios = TOWN_SCENARIOS[town]
        scenario_steps = {scenario: 0 for scenario in allowed_scenarios}
        for row in current_rows:
            setting = str(row.get("setting_id", "unknown/unknown/unknown"))
            scenario = str(row.get("scenario") or setting.split("/", 1)[0]).lower()
            if scenario in scenario_steps:
                scenario_steps[scenario] += max(
                    0, int(_number(row, "episode_length") or 0)
                )
        exact = False
        fragment = log_dir.parent / "checkpoints" / f"update_{update_index:03d}" / "fragments" / f"{town}.json"
        fragment_progress = _fragment_scenario_steps(fragment, allowed_scenarios)
        if fragment_progress is not None:
            current_step, scenario_steps = fragment_progress
            exact = True
        displayed_step = min(current_step, target) if target > 0 else current_step
        in_flight_steps = max(0, displayed_step - sum(scenario_steps.values()))
        display_scenario_steps = {
            training_display_scenario_id(scenario): steps
            for scenario, steps in scenario_steps.items()
        }
        for scenario, steps in display_scenario_steps.items():
            scenario_progress[scenario] = {"steps": steps, "exact": exact}
        workers.append({
            "town": town,
            "target_steps": target,
            "current_step": displayed_step,
            "current_episodes": len(current_rows),
            "scenario_steps": display_scenario_steps,
            "in_flight_steps": in_flight_steps,
            "exact": exact,
            "episodes": public_town,
        })
        offset += target
    scenarios, _all_run_scenario_steps = _scenario_series(public_episodes)
    experiment = _experiment_progress(
        log_dir=log_dir,
        state=state,
        workers=workers,
        epochs=epochs,
        validation=validation,
        raw_episodes=raw_episodes,
        events=events,
        total_updates=total_updates,
        configured_steps_per_update=steps_per_update,
    )
    return {
        "log_dir": str(log_dir.resolve()),
        "episode_sources": episode_sources,
        "epochs_file": epochs_path.name,
        "run_status": str(state.get("status", "unknown")),
        "assessment": _assessment(raw_episodes, epochs),
        "workers": workers,
        "scenarios": scenarios,
        "scenario_progress": scenario_progress,
        "episodes": public_episodes,
        "epochs": [
            {
                "environment_steps": _number(row, "environment_steps"),
                "policy_loss": _number(row, "policy_loss"),
                "value_loss": _number(row, "value_loss"),
                "approx_kl": _number(row, "approx_kl"),
                "clip_fraction": _number(row, "clip_fraction"),
            }
            for row in epochs
        ],
        "validation": validation,
        "experiment": experiment,
    }


def _handler(
    log_dir: Path,
    experiment_name: str,
    total_updates: int | None = None,
    steps_per_update: int | None = None,
) -> type[BaseHTTPRequestHandler]:
    rendered_html = HTML.replace(
        "__EXPERIMENT_NAME__", escape(experiment_name, quote=True)
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                body = rendered_html
                content_type = "text/html; charset=utf-8"
            elif path == "/api/data":
                body = json.dumps(
                    _payload(
                        log_dir,
                        total_updates=total_updates,
                        steps_per_update=steps_per_update,
                    ),
                    allow_nan=False,
                ).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--experiment-name", default="PPO V2 Live Training")
    parser.add_argument("--total-updates", type=int, default=0)
    parser.add_argument("--steps-per-update", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(
            args.log_dir,
            args.experiment_name,
            total_updates=args.total_updates or None,
            steps_per_update=args.steps_per_update or None,
        ),
    )
    print(f"PPO V2 dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"Telemetry: {args.log_dir.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
