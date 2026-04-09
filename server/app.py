"""FastAPI app for the Pollution Exposure Minimizer Environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install project dependencies before running the server."
    ) from exc

try:
    from ..models import GradeRequest, GradeResponse, PollutionAction, PollutionObservation
    from .baseline import get_baseline_summary
    from .grader import grade_request
    from .pollution_exposure_minimizer_environment import PollutionExposureMinimizerEnvironment
    from .tasks import TASK_ORDER, list_task_summaries
except ImportError:  # pragma: no cover
    from models import GradeRequest, GradeResponse, PollutionAction, PollutionObservation
    from server.baseline import get_baseline_summary
    from server.grader import grade_request
    from server.pollution_exposure_minimizer_environment import PollutionExposureMinimizerEnvironment
    from server.tasks import TASK_ORDER, list_task_summaries


app = create_app(
    PollutionExposureMinimizerEnvironment,
    PollutionAction,
    PollutionObservation,
    env_name="pollution-exposure-minimizer-environment",
    max_concurrent_envs=8,
)

class DemoResetRequest(BaseModel):
    task_id: str | None = Field(default=None)
    seed: int | None = Field(default=None)


class DemoStepRequest(BaseModel):
    session_id: str
    action_type: Literal["move", "wait"]
    target_node_id: str | None = None
    mode: Literal["walk", "bus", "metro"] | None = None


class DemoSessionRequest(BaseModel):
    session_id: str


@dataclass
class DemoSession:
    env: PollutionExposureMinimizerEnvironment
    trail: list[str] = field(default_factory=list)
    observations: list[PollutionObservation] = field(default_factory=list)


_demo_sessions: dict[str, DemoSession] = {}
_demo_lock = Lock()


DEMO_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pollution Exposure Minimizer</title>
  <style>
    :root{--bg:#eef4ea;--card:#ffffffdd;--ink:#142118;--muted:#587063;--line:#d8e3db;--accent:#1f7a63;--trail:#2b7da1;--baseline:#c26b38}
    *{box-sizing:border-box}body{margin:0;font-family:Georgia,serif;color:var(--ink);background:linear-gradient(180deg,#f6faf4,#e9f0e6)}
    .page{width:min(1520px,calc(100vw - 24px));margin:0 auto;padding:18px;display:grid;grid-template-columns:minmax(0,1fr);gap:20px}
    .hero,.panel{background:var(--card);border:1px solid #fff;border-radius:22px;box-shadow:0 16px 40px rgba(28,45,35,.08)}
    .hero{padding:20px 22px;background:linear-gradient(135deg,#185948,#2e7d6c);color:#f7fbf6}
    .hero h1{margin:0;font-size:2.15rem;letter-spacing:-.03em}.hero p{margin:8px 0 0;max-width:68ch;line-height:1.45;color:#eef7f2}
    .layout{position:relative;display:grid;grid-template-columns:clamp(320px,26vw,400px) minmax(0,1fr);gap:28px;align-items:start;overflow:hidden;isolation:isolate}.sidebar{position:relative;z-index:2;display:grid;gap:16px;min-width:0;align-self:start}.main{position:relative;z-index:1;display:grid;gap:16px;min-width:0;width:100%;overflow:hidden;align-self:start}.panel{padding:16px}
    .muted{color:var(--muted)}.title{font-size:.84rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
    select,button{font:inherit}.row{display:grid;grid-template-columns:1fr;gap:10px}.btn{border:none;border-radius:14px;padding:11px 12px;cursor:pointer}
    .btn-primary{background:var(--accent);color:#fff}.btn-soft{background:#dff1ea;color:#155545}.btn-ghost{background:#eef3f0;color:var(--ink)}
    .stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}.stat{padding:12px;border-radius:16px;background:#fbfcfb;border:1px solid var(--line)}
    .stat span{display:block;font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.stat strong{display:block;margin-top:5px}
    #actions{display:grid;gap:10px;max-height:420px;overflow:auto}.action{width:100%;text-align:left;padding:12px;border:none;border-radius:16px;background:#fbfcfb;border:1px solid var(--line);cursor:pointer}
    .action strong{display:block}.action small{display:block;margin-top:6px;color:#91542b}
    .map-panel{position:relative;z-index:1;padding:0;max-width:100%;overflow:hidden;isolation:isolate;box-shadow:0 12px 28px rgba(28,45,35,.06)}.map-head{padding:18px 18px 0}.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .chip{padding:8px 12px;border-radius:999px;background:#e3f0ea;color:#1f624f;font-size:.86rem}.map-shell{position:relative;padding:16px;min-width:0;max-width:100%;overflow:hidden}
    #map{width:100%;max-width:100%;height:clamp(240px,30vw,340px);min-height:240px;display:block;overflow:hidden;background:radial-gradient(circle at top left,#fff4df,transparent 22%),linear-gradient(180deg,#fff,#f4f8f3);border-radius:22px;border:1px solid var(--line)}
    .legend{position:absolute;right:24px;bottom:24px;background:#fffffff0;border:1px solid var(--line);border-radius:16px;padding:10px 12px;font-size:.82rem;max-width:220px}
    .legend div{margin-top:6px;display:flex;align-items:center;gap:8px;color:var(--muted)}.sw{width:12px;height:12px;border-radius:999px;display:inline-block}
    .summary{position:relative;z-index:2;display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr) minmax(0,1.8fr);gap:16px;min-width:0}.summary .panel{height:100%}
    pre{margin:0;background:#132018;color:#dbf3e8;padding:14px;border-radius:16px;overflow:auto;max-height:260px;font-size:.8rem;white-space:pre-wrap;word-break:break-word}
    #message{display:none;padding:12px 14px;border-radius:14px;background:#e7f3ee;color:#175746}#message.show{display:block}.error{background:#f8e7e0!important;color:#9d4d28!important}
    a{color:inherit}.links{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:.92rem;margin-top:12px}
    #taskDescription{white-space:pre-line;overflow-wrap:anywhere}
    @media (max-width:1180px){.layout,.summary{grid-template-columns:1fr}.layout{gap:16px}} @media (max-width:720px){.stats{grid-template-columns:1fr}.page{width:min(100vw - 12px,1520px);padding:14px}#map{height:260px;min-height:220px}}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Pollution Exposure Minimizer</h1>
      <p>This is meant to model a real decision problem: choosing how to commute through a polluted city while trading off exposure, travel time, and route structure.</p>
      <p>Interactive Delhi commute demo for the OpenEnv benchmark. Reset any task, step through legal actions, or let the built-in demo agent traverse the AQI graph while the route trail updates live.</p>
    </section>
    <section class="layout">
      <aside class="sidebar">
        <section class="panel">
          <div id="message"></div>
          <p class="title">Scenario</p>
          <select id="taskSelect"></select>
          <div class="row">
            <button id="resetBtn" class="btn btn-primary">Reset</button>
            <button id="autoBtn" class="btn btn-soft">Auto Step</button>
            <button id="runBtn" class="btn btn-ghost">Run Agent</button>
            <button id="stopBtn" class="btn btn-ghost">Stop</button>
          </div>
          <div class="stats" style="margin-top:14px">
            <div class="stat"><span>Current AQI</span><strong id="aqiStat">-</strong></div>
            <div class="stat"><span>Current Hour</span><strong id="hourStat">-</strong></div>
            <div class="stat"><span>Exposure</span><strong id="exposureStat">-</strong></div>
            <div class="stat"><span>Travel Time</span><strong id="timeStat">-</strong></div>
            <div class="stat"><span>Steps Left</span><strong id="stepsStat">-</strong></div>
            <div class="stat"><span>Score</span><strong id="scoreStat">-</strong></div>
          </div>
          <div class="links">
            <a href="/tasks" target="_blank" rel="noreferrer">Tasks</a>
            <a href="/baseline" target="_blank" rel="noreferrer">Baseline</a>
            <a href="/grader" target="_blank" rel="noreferrer">Grader</a>
            <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
          </div>
        </section>
        <section class="panel">
          <p class="title">Legal Actions</p>
          <div id="actions"></div>
        </section>
      </aside>
      <main class="main">
        <section class="panel map-panel">
          <div class="map-head">
            <h2 id="taskTitle" style="margin:0">Loading task...</h2>
            <p id="taskDescription" class="muted" style="margin:8px 0 0;line-height:1.55">Preparing city graph.</p>
            <div class="chips">
              <div class="chip" id="currentChip">Current: -</div>
              <div class="chip" id="destinationChip">Destination: -</div>
              <div class="chip" id="difficultyChip">Difficulty: -</div>
            </div>
          </div>
          <div class="map-shell">
            <svg id="map" viewBox="0 0 980 620" preserveAspectRatio="xMidYMid meet"></svg>
            <div class="legend">
              <strong>AQI & Path</strong>
              <div><span class="sw" style="background:#58b368"></span>Cleaner node</div>
              <div><span class="sw" style="background:#e7bd42"></span>Elevated AQI</div>
              <div><span class="sw" style="background:#d96539"></span>High AQI</div>
              <div><span class="sw" style="background:#2b7da1"></span>Live trail</div>
              <div><span class="sw" style="background:#c26b38"></span>Baseline path</div>
            </div>
          </div>
        </section>
      </main>
    </section>
    <section class="summary">
      <section class="panel"><p class="title">Episode Summary</p><p id="summaryText" class="muted" style="line-height:1.55;margin:0">Reset a task to begin.</p></section>
      <section class="panel"><p class="title">Reference Route</p><p id="referenceText" class="muted" style="line-height:1.55;margin:0">Baseline and oracle paths will appear here.</p></section>
      <section class="panel"><p class="title">Raw State</p><pre id="jsonPanel">{}</pre></section>
    </section>
  </div>
  <script>
    const state={sessionId:null,observation:null,envState:null,trail:[],baseline:null,autorun:false};
    const el=id=>document.getElementById(id);
    const msg=el("message");
    function setMsg(text,kind){msg.textContent=text;if(!text){msg.className="";return}msg.className=`show ${kind||""}`.trim();}
    async function api(url,options={}){const r=await fetch(url,{headers:{"Content-Type":"application/json"},...options});if(!r.ok){throw new Error(await r.text()||`${r.status} ${r.statusText}`)}return r.json()}
    function esc(t){return String(t).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;")}
    function aqiColor(v){if(v<150)return "#58b368";if(v<180)return "#e7bd42";return "#d96539"}
    function positions(nodes){const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),padX=105,padY=78,w=720,h=360;const out={};nodes.forEach(n=>{const nx=maxX===minX ? .5 : (n.x-minX)/(maxX-minX);const ny=maxY===minY ? .5 : (n.y-minY)/(maxY-minY);out[n.node_id]={x:padX+nx*w,y:padY+(1-ny)*h}});return out}
    function edgeLookup(path){const s=new Set();for(let i=0;i<path.length-1;i++){s.add(`${path[i]}::${path[i+1]}`);s.add(`${path[i+1]}::${path[i]}`)}return s}
    function actionBody(option){return {action_type:option.action_type,target_node_id:option.target_node_id,mode:option.mode}}
    function renderMap(){if(!state.observation){el("map").innerHTML="";return}const obs=state.observation,pos=positions(obs.graph_nodes),trail=edgeLookup(state.trail),baseline=edgeLookup(state.baseline?state.baseline.baseline_path:[]);const edges=obs.graph_edges.map(edge=>{const a=pos[edge.source_node_id],b=pos[edge.target_node_id],trailOn=trail.has(`${edge.source_node_id}::${edge.target_node_id}`),baseOn=baseline.has(`${edge.source_node_id}::${edge.target_node_id}`),stroke=trailOn?"#2b7da1":baseOn?"#c26b38":"rgba(22,35,28,.16)",width=trailOn?7:baseOn?4:2.5,dash=trailOn?"0":baseOn?"10 10":"0";return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${stroke}" stroke-width="${width}" stroke-linecap="round" stroke-dasharray="${dash}" />`}).join("");const nodes=obs.graph_nodes.map(node=>{const p=pos[node.node_id],current=node.node_id===obs.current_node_id,dest=node.node_id===obs.destination_node_id,r=current?16:dest?14:11,ring=current?"#1f7a63":dest?"#093f5e":"rgba(20,33,24,.2)";return `<g><circle cx="${p.x}" cy="${p.y}" r="${r+5}" fill="rgba(255,255,255,.95)" /><circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${aqiColor(node.current_aqi)}" stroke="${ring}" stroke-width="${current||dest?3.5:2}" /><text x="${p.x}" y="${p.y-r-10}" text-anchor="middle" style="font:600 10px Georgia,serif;fill:#142118">${esc(node.label)}</text><text x="${p.x}" y="${p.y+4}" text-anchor="middle" style="font:8px monospace;fill:rgba(20,33,24,.75)">AQI ${Math.round(node.current_aqi)}</text></g>`}).join("");const points=state.trail.map((id,idx)=>{const p=pos[id];return `<circle cx="${p.x}" cy="${p.y}" r="4" fill="${idx===state.trail.length-1?"#1f7a63":"#2b7da1"}" opacity="${Math.max(.35,(idx+1)/state.trail.length)}" />`}).join("");el("map").innerHTML=`<rect x="0" y="0" width="980" height="620" fill="transparent"></rect>${edges}${points}${nodes}`;}
    function renderActions(){const obs=state.observation;if(!obs){el("actions").innerHTML="";return}if(obs.done){el("actions").innerHTML='<div class="action"><strong>Episode complete</strong><div class="muted" style="margin-top:6px">Reset to start again.</div></div>';return}el("actions").innerHTML=obs.legal_actions.map((a,i)=>`<button class="action" onclick="window.stepByIndex(${i})"><strong>${esc(a.action_type==="wait"?"Wait in place":`${a.target_label} via ${a.mode}`)}</strong><div class="muted" style="margin-top:6px;line-height:1.45">${esc(a.description)}</div><small>Exposure ${a.estimated_exposure.toFixed(2)} · ${a.estimated_time_minutes} min</small></button>`).join("")}
    function renderSummary(){const obs=state.observation,env=state.envState;if(!obs||!env)return;el("taskTitle").textContent=obs.task_name;el("taskDescription").textContent=obs.prompt;el("currentChip").textContent=`Current: ${obs.current_node_label}`;el("destinationChip").textContent=`Destination: ${obs.destination_node_label}`;el("difficultyChip").textContent=`Difficulty: ${obs.difficulty}`;el("aqiStat").textContent=Math.round(obs.current_node_aqi);el("hourStat").textContent=`${obs.current_hour.toFixed(2)} h`;el("exposureStat").textContent=obs.cumulative_exposure.toFixed(2);el("timeStat").textContent=`${obs.cumulative_time_minutes} min`;el("stepsStat").textContent=`${obs.steps_remaining} left`;el("scoreStat").textContent=env.episode_score==null?"-":env.episode_score.toFixed(3);el("summaryText").textContent=obs.final_summary||`Trail: ${state.trail.join(" → ")}. Last action: ${env.last_action_summary||"Waiting for first move."}`;el("referenceText").textContent=state.baseline?`Baseline: ${state.baseline.baseline_path.join(" → ")}. Oracle: ${state.baseline.oracle_path.join(" → ")}.`:"Reference paths unavailable.";el("jsonPanel").textContent=JSON.stringify({session_id:state.sessionId,observation:obs,state:env,trail:state.trail},null,2)}
    async function hydrate(payload){state.sessionId=payload.session_id;state.observation=payload.observation;state.envState=payload.state;state.trail=payload.trail||[];state.baseline=payload.baseline||null;renderMap();renderActions();renderSummary()}
    async function resetSession(){setMsg("Resetting environment...");try{await hydrate(await api("/demo/reset",{method:"POST",body:JSON.stringify({task_id:el("taskSelect").value||null})}));setMsg("")}catch(err){setMsg(err.message,"error")}}
    async function stepAction(action){try{await hydrate(await api("/demo/step",{method:"POST",body:JSON.stringify({session_id:state.sessionId,...action})}))}catch(err){setMsg(err.message,"error")}}
    async function autoStep(){try{await hydrate(await api("/demo/auto-step",{method:"POST",body:JSON.stringify({session_id:state.sessionId})}))}catch(err){setMsg(err.message,"error")}}
    async function runAgent(){state.autorun=true;while(state.autorun&&state.observation&&!state.observation.done){await autoStep();await new Promise(r=>setTimeout(r,650))}}
    window.stepByIndex=async index=>{state.autorun=false;const action=state.observation?.legal_actions?.[index];if(action)await stepAction(actionBody(action))}
    el("resetBtn").addEventListener("click",async()=>{state.autorun=false;await resetSession()});el("autoBtn").addEventListener("click",async()=>{state.autorun=false;await autoStep()});el("runBtn").addEventListener("click",async()=>{if(!state.observation||state.observation.done)await resetSession();await runAgent()});el("stopBtn").addEventListener("click",()=>{state.autorun=false;setMsg("Autorun stopped.")});el("taskSelect").addEventListener("change",async()=>{state.autorun=false;await resetSession()});
    (async()=>{try{const tasks=await api("/tasks");el("taskSelect").innerHTML=tasks.map(task=>`<option value="${task.task_id}">${task.name} · ${task.difficulty}</option>`).join("");await resetSession()}catch(err){setMsg(err.message,"error")}})();
  </script>
</body>
</html>
"""


def _distance(node_a: str, node_b: str, observation: PollutionObservation) -> float:
    nodes = {node.node_id: node for node in observation.graph_nodes}
    first = nodes[node_a]
    second = nodes[node_b]
    return abs(first.x - second.x) + abs(first.y - second.y)


def _score_demo_action(
    observation: PollutionObservation,
    action_index: int,
    trail: list[str],
) -> float:
    option = observation.legal_actions[action_index]
    current_distance = _distance(
        observation.current_node_id,
        observation.destination_node_id,
        observation,
    )
    if option.action_type == "wait":
        repeat_wait = 14.0 if trail[-2:] == [observation.current_node_id, observation.current_node_id] else 0.0
        return (
            option.estimated_exposure * 0.92
            + option.estimated_time_minutes * 4.8
            + current_distance * 36.0
            + 28.0
            + repeat_wait
        )

    target_id = option.target_node_id or observation.current_node_id
    target_distance = _distance(target_id, observation.destination_node_id, observation)
    progress_bonus = (current_distance - target_distance) * 18.0
    revisit_penalty = 0.0
    if target_id in trail[-4:]:
        revisit_penalty += 24.0
    if len(trail) >= 2 and target_id == trail[-2]:
        revisit_penalty += 34.0
    destination_bonus = -240.0 if target_id == observation.destination_node_id else 0.0
    mode_bonus = {"metro": -6.0, "bus": -2.2, "walk": 0.0}.get(option.mode, 0.0)

    return (
        option.estimated_exposure * 0.85
        + option.estimated_time_minutes * 2.55
        + target_distance * 26.0
        + revisit_penalty
        + destination_bonus
        + mode_bonus
        - progress_bonus
    )


def _pack_demo_response(session_id: str, session: DemoSession) -> dict:
    observation = session.observations[-1]
    return {
        "session_id": session_id,
        "observation": observation.model_dump(mode="json"),
        "state": session.env.state.model_dump(mode="json"),
        "trail": session.trail,
        "baseline": get_baseline_summary(observation.task_id).model_dump(mode="json"),
    }


def _load_demo_session(session_id: str) -> DemoSession:
    session = _demo_sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown demo session. Reset the task to start again.",
        )
    return session


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)


@app.post("/demo/reset")
def demo_reset(request: DemoResetRequest | None = Body(default=None)) -> dict:
    request = request or DemoResetRequest()
    env = PollutionExposureMinimizerEnvironment()
    observation = env.reset(task_id=request.task_id, seed=request.seed)
    session_id = str(uuid4())
    session = DemoSession(
        env=env,
        trail=[observation.current_node_id],
        observations=[observation],
    )
    with _demo_lock:
        _demo_sessions[session_id] = session
    return _pack_demo_response(session_id, session)


@app.post("/demo/step")
def demo_step(request: DemoStepRequest = Body(...)) -> dict:
    with _demo_lock:
        session = _load_demo_session(request.session_id)
        observation = session.env.step(
            PollutionAction(
                action_type=request.action_type,
                target_node_id=request.target_node_id,
                mode=request.mode,
                rationale="Custom demo action",
            )
        )
        session.trail.append(observation.current_node_id)
        session.observations.append(observation)
        return _pack_demo_response(request.session_id, session)


@app.post("/demo/auto-step")
def demo_auto_step(request: DemoSessionRequest = Body(...)) -> dict:
    with _demo_lock:
        session = _load_demo_session(request.session_id)
        observation = session.observations[-1]
        if observation.done:
            return _pack_demo_response(request.session_id, session)
        best_index = min(
            range(len(observation.legal_actions)),
            key=lambda idx: _score_demo_action(observation, idx, session.trail),
        )
        option = observation.legal_actions[best_index]
        next_observation = session.env.step(
            PollutionAction(
                action_type=option.action_type,
                target_node_id=option.target_node_id,
                mode=option.mode,
                rationale="Deterministic demo agent",
            )
        )
        session.trail.append(next_observation.current_node_id)
        session.observations.append(next_observation)
        return _pack_demo_response(request.session_id, session)


@app.get("/tasks")
def tasks() -> list[dict]:
    return [task.model_dump() for task in list_task_summaries()]


@app.get("/baseline")
def baseline(task_id: str | None = None) -> dict | list[dict]:
    if task_id:
        return get_baseline_summary(task_id).model_dump()
    return [get_baseline_summary(task).model_dump() for task in TASK_ORDER]


@app.get("/grader")
def grader_overview(task_id: str | None = None) -> dict:
    requested_tasks = [task_id] if task_id else list(TASK_ORDER)
    references = {task: get_baseline_summary(task).model_dump() for task in requested_tasks}
    return {
        "score_range": [0.01, 0.99],
        "formula": "score = clamp((baseline_cost - agent_cost) / (baseline_cost - oracle_cost), 0.01, 0.99); agent_cost includes exposure, time, wait penalties, failure penalty, and remaining-distance penalty.",
        "references": references,
    }


@app.get("/validate")
def validate_overview() -> dict:
    tasks = list_task_summaries()
    references = {task.task_id: get_baseline_summary(task.task_id).model_dump() for task in tasks}
    score_range = [0.01, 0.99]
    return {
        "task_count": len(tasks),
        "tasks_with_graders": sum(1 for task in tasks if task.grader),
        "all_tasks_have_graders": all(task.grader for task in tasks),
        "score_range": score_range,
        "task_ids": [task.task_id for task in tasks],
        "references": references,
    }


@app.post("/grader", response_model=GradeResponse)
def grader(request: GradeRequest = Body(...)) -> GradeResponse:
    baseline = get_baseline_summary(request.task_id)
    return grade_request(
        request=request,
        baseline_cost=baseline.baseline_cost,
        oracle_cost=baseline.oracle_cost,
    )


def main(host: str = "0.0.0.0", port: int = 7680) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
