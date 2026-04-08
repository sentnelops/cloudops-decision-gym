"""
CloudOps Decision Gym — Premium Gradio UI
Deployable to Hugging Face Spaces.
Run locally: python app.py  →  http://localhost:7860
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel as _BaseModel

from env.actions import Action as InternalAction
from env.environment import CloudOpsEnv
from graders.ec2_grader import grade_episode_detailed
from openenv_models import Action, Observation

# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------

SCENARIOS = {
    "easy": {
        "icon": "🟢",
        "title": "Dev Instance — Cost Waste Detected",
        "description": "An m5.2xlarge dev instance running at only 7% CPU. "
                       "It is overprovisioned by ~10× with no security concerns.",
        "goal": "Downsize the instance to eliminate cost waste safely.",
        "difficulty": "Easy",
        "border": "#22c55e",
        "bg": "#f0fdf4",
    },
    "medium": {
        "icon": "🟡",
        "title": "Dev Instance — Security Conflict",
        "description": "An m5.xlarge dev instance at 12% CPU, publicly internet-facing "
                       "with IMDSv1 enabled and SSH open to 0.0.0.0/0.",
        "goal": "Fix security vulnerabilities before optimizing cost. Order matters.",
        "difficulty": "Medium",
        "border": "#f59e0b",
        "bg": "#fffbeb",
    },
    "hard": {
        "icon": "🔴",
        "title": "Production Instance at Risk",
        "description": "A prod m5.2xlarge at 35% CPU with all security vulnerabilities exposed. "
                       "Production environment amplifies all penalties by 1.5×.",
        "goal": "Remediate security first, then reduce cost — without breaking production.",
        "difficulty": "Hard",
        "border": "#ef4444",
        "bg": "#fef2f2",
    },
}

SCORE_WEIGHTS = {
    "cost_reduction": ("Cost Optimization", 0.25),
    "safety_maintained": ("Security Handling", 0.25),
    "sequence_correctness": ("Decision Sequence", 0.25),
    "no_risky_actions": ("Risk Management", 0.15),
    "efficiency": ("Efficiency", 0.10),
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── Global ────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1200px !important; margin: 0 auto; }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero-block { text-align: center; padding: 28px 0 12px 0; }
.hero-block h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 6px 0; }
.hero-block p  { font-size: 1.05rem; color: #6b7280 !important; margin: 0; }

/* ── Panels ─────────────────────────────────────────────────────────────── */
.panel {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
    color: #111827 !important;
}
.panel-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af !important;
    margin: 0 0 14px 0;
}

/* ── State rows ─────────────────────────────────────────────────────────── */
.state-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f3f4f6 !important;
    font-size: 0.88rem;
}
.state-row:last-child { border-bottom: none !important; }
.state-key { color: #6b7280 !important; }
.state-val { font-weight: 600; font-family: monospace; color: #111827 !important; }

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-green  { background: #dcfce7 !important; color: #15803d !important; }
.badge-yellow { background: #fef9c3 !important; color: #a16207 !important; }
.badge-red    { background: #fee2e2 !important; color: #b91c1c !important; }
.badge-blue   { background: #dbeafe !important; color: #1d4ed8 !important; }
.badge-gray   { background: #f3f4f6 !important; color: #374151 !important; }

/* ── Action timeline entries ────────────────────────────────────────────── */
.action-entry {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.875rem;
    font-family: monospace;
}
.action-good    { background: #f0fdf4 !important; }
.action-warn    { background: #fffbeb !important; }
.action-bad     { background: #fef2f2 !important; }
.action-neutral { background: #f9fafb !important; }
.action-icon    { font-size: 1rem; min-width: 20px; }
.action-name    { flex: 1; font-weight: 600; color: #111827 !important; }
.action-reward  { font-weight: 700; min-width: 56px; text-align: right; }
.reward-pos     { color: #16a34a !important; }
.reward-neg     { color: #dc2626 !important; }
.action-cost    { color: #6b7280 !important; font-size: 0.78rem; margin-left: 4px; }

/* ── Score bars ─────────────────────────────────────────────────────────── */
.score-header {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 18px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.score-row { margin-bottom: 12px; }
.score-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    color: #374151 !important;
    margin-bottom: 4px;
}
.score-bar-track {
    height: 8px;
    background: #f3f4f6 !important;
    border-radius: 4px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* ── Reasoning log ──────────────────────────────────────────────────────── */
.reasoning-log {
    font-size: 0.875rem;
    line-height: 1.7;
    color: #374151 !important;
}
.reasoning-step {
    padding: 6px 0;
    border-bottom: 1px dashed #e5e7eb !important;
}
.reasoning-step:last-child { border-bottom: none !important; }
.step-label {
    font-weight: 600;
    color: #6366f1 !important;
    margin-right: 6px;
}

/* ── Status banner ──────────────────────────────────────────────────────── */
.status-banner {
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.status-ready   { background: #f0f9ff !important; color: #0369a1 !important; border: 1px solid #bae6fd !important; }
.status-running { background: #fffbeb !important; color: #92400e !important; border: 1px solid #fde68a !important; }
.status-done-ok { background: #f0fdf4 !important; color: #166534 !important; border: 1px solid #bbf7d0 !important; }
.status-done-to { background: #fef2f2 !important; color: #991b1b !important; border: 1px solid #fecaca !important; }

/* ── Why section ────────────────────────────────────────────────────────── */
.why-section {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 0.92rem;
    line-height: 1.7;
    margin-top: 8px;
}
.why-section strong { color: #f1f5f9 !important; }

/* ── Controls row ───────────────────────────────────────────────────────── */
.controls-row { align-items: flex-end; }
"""

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def empty_state() -> dict[str, Any]:
    return {
        "env": None,
        "obs": None,
        "initial_state": None,
        "actions_taken": [],
        "step_history": [],
        "done": False,
        "scenario": "easy",
        "grader_score": 0.0,
        "score_detail": {},
    }


# ---------------------------------------------------------------------------
# Reasoning engine
# ---------------------------------------------------------------------------

def get_reasoning(obs: Observation) -> tuple[str, str]:
    if obs.imds_version == "v1" or obs.ssh_open:
        issues = []
        if obs.imds_version == "v1":
            issues.append("IMDSv1 active (SSRF credential theft risk)")
        if obs.ssh_open:
            issues.append("SSH port 22 open to 0.0.0.0/0")
        return (
            f"🔴 Security violations: {' | '.join(issues)}. "
            f"Must remediate before any sizing change.",
            "FIX_SECURITY",
        )
    if obs.cpu_p95 > 60.0:
        return (
            f"🟡 CPU p95 = {obs.cpu_p95:.1f}% exceeds 60% safety threshold. "
            f"Downsizing would risk instability. Holding.",
            "NOOP",
        )
    if obs.memory_avg > 70.0:
        return (
            f"🟡 Memory = {obs.memory_avg:.1f}% exceeds 70% safety threshold. "
            f"Downsizing would risk OOM. Holding.",
            "NOOP",
        )
    if obs.cpu_avg < 20.0 and obs.memory_avg < 30.0:
        return (
            f"🟢 Security clear. CPU avg = {obs.cpu_avg:.1f}%, Memory = {obs.memory_avg:.1f}%. "
            f"Instance is overprovisioned. Safe to DOWNSIZE.",
            "DOWNSIZE",
        )
    if obs.cpu_avg < 50.0 and obs.cpu_p95 <= 60.0:
        return (
            f"🟢 Security clear. CPU avg = {obs.cpu_avg:.1f}%, p95 = {obs.cpu_p95:.1f}%. "
            f"Moderate load, within safe limits. DOWNSIZE to reduce cost.",
            "DOWNSIZE",
        )
    return (
        f"⚪ No clear optimization needed. "
        f"CPU avg = {obs.cpu_avg:.1f}%, p95 = {obs.cpu_p95:.1f}%.",
        "NOOP",
    )


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{text}</span>'


def render_scenario_card(scenario: str) -> str:
    meta = SCENARIOS[scenario]
    return f"""
<div style="background:{meta['bg']};border-left:4px solid {meta['border']};
            border-radius:8px;padding:16px 20px;margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:1.3rem;">{meta['icon']}</span>
    <span style="font-size:1.05rem;font-weight:700;color:#111827;">{meta['title']}</span>
    <span style="margin-left:auto;font-size:0.75rem;font-weight:600;
                 color:{meta['border']};text-transform:uppercase;letter-spacing:0.05em;">
      {meta['difficulty']}
    </span>
  </div>
  <p style="margin:0 0 8px 0;font-size:0.875rem;color:#374151;">{meta['description']}</p>
  <p style="margin:0;font-size:0.875rem;">
    <strong style="color:#111827;">Goal:</strong>
    <span style="color:#374151;"> {meta['goal']}</span>
  </p>
</div>"""


def _cpu_badge(cpu: float) -> str:
    if cpu < 20:
        return _badge(f"{cpu:.1f}% 🟢 Low", "green")
    if cpu < 60:
        return _badge(f"{cpu:.1f}% 🟡 Moderate", "yellow")
    return _badge(f"{cpu:.1f}% 🔴 High", "red")


def _cost_badge(cost: float) -> str:
    if cost < 100:
        return _badge(f"${cost:.2f} 🟢", "green")
    if cost < 200:
        return _badge(f"${cost:.2f} 🟡", "yellow")
    return _badge(f"${cost:.2f} 🔴 High", "red")


def _bool_badge(val: bool, true_bad: bool = True) -> str:
    if val:
        cls = "red" if true_bad else "green"
        icon = "🔴" if true_bad else "🟢"
    else:
        cls = "green" if true_bad else "gray"
        icon = "🟢" if true_bad else "—"
    return _badge(f"{val} {icon}", cls)


def _imds_badge(ver: str) -> str:
    if ver == "v1":
        return _badge("v1 🔴 Vulnerable", "red")
    return _badge("v2 🟢 Secure", "green")


def _env_badge(env: str) -> str:
    if env == "prod":
        return _badge("prod ⚠️", "yellow")
    return _badge("dev", "gray")


def render_infra_state(obs: Observation | None) -> str:
    if obs is None:
        return '<div class="panel"><p class="panel-title">Infrastructure State</p><p style="color:#9ca3af;font-size:0.875rem;">Select a scenario and click Reset.</p></div>'

    rows = [
        ("Instance Type", _badge(obs.instance_type, "blue")),
        ("CPU Avg", _cpu_badge(obs.cpu_avg)),
        ("CPU P95", _cpu_badge(obs.cpu_p95)),
        ("Memory Avg", _badge(f"{obs.memory_avg:.1f}%", "green" if obs.memory_avg < 70 else "red")),
        ("Monthly Cost", _cost_badge(obs.monthly_cost)),
        ("Environment", _env_badge(obs.environment)),
        ("Internet Facing", _bool_badge(obs.internet_facing, true_bad=True)),
        ("SSH Open", _bool_badge(obs.ssh_open, true_bad=True)),
        ("IMDS Version", _imds_badge(obs.imds_version)),
        ("Security Status", _badge("✅ SECURE", "green") if obs.imds_version == "v2" and not obs.ssh_open else _badge("🚨 INSECURE", "red")),
    ]

    rows_html = "".join(
        f'<div class="state-row"><span class="state-key">{k}</span>'
        f'<span class="state-val">{v}</span></div>'
        for k, v in rows
    )

    return f'<div class="panel"><p class="panel-title">🖥️ Infrastructure State</p>{rows_html}</div>'


def render_reasoning(step_history: list[dict]) -> str:
    if not step_history:
        return '<div class="panel"><p class="panel-title">🧠 Agent Reasoning</p><p style="color:#9ca3af;font-size:0.875rem;">Reasoning will appear as the agent takes actions.</p></div>'

    entries = []
    for h in step_history:
        entries.append(
            f'<div class="reasoning-step">'
            f'<span class="step-label">[Step {h["step"]}]</span>'
            f'<span style="color:#111827;">{h["reasoning"]}</span>'
            f'</div>'
        )

    return (
        f'<div class="panel">'
        f'<p class="panel-title">🧠 Agent Reasoning</p>'
        f'<div class="reasoning-log" style="color:#111827;">{"".join(entries)}</div>'
        f'</div>'
    )


def _action_entry(step: int, action: str, reward: float, cost_after: float) -> str:
    if reward > 0.3:
        cls, icon = "action-good", "✅"
    elif reward > 0:
        cls, icon = "action-good", "✔️"
    elif reward > -0.2:
        cls, icon = "action-warn", "⚠️"
    else:
        cls, icon = "action-bad", "✖️"

    reward_cls = "reward-pos" if reward >= 0 else "reward-neg"
    sign = "+" if reward >= 0 else ""

    action_labels = {
        "FIX_SECURITY": "FIX_SECURITY",
        "DOWNSIZE": "DOWNSIZE",
        "UPSIZE": "UPSIZE",
        "NOOP": "NOOP",
    }

    return (
        f'<div class="action-entry {cls}">'
        f'<span class="action-icon">{icon}</span>'
        f'<span style="color:#6b7280;font-size:0.78rem;min-width:28px;">#{step}</span>'
        f'<span class="action-name">{action_labels.get(action, action)}</span>'
        f'<span class="action-reward {reward_cls}">{sign}{reward:.3f}</span>'
        f'<span class="action-cost">${cost_after:.2f}/mo</span>'
        f'</div>'
    )


def render_actions(step_history: list[dict]) -> str:
    if not step_history:
        return '<div class="panel"><p class="panel-title">⚡ Actions Timeline</p><p style="color:#9ca3af;font-size:0.875rem;">Actions will appear here as the agent runs.</p></div>'

    total = sum(h["reward"] for h in step_history)
    sign = "+" if total >= 0 else ""
    entries_html = "".join(
        _action_entry(h["step"], h["action"], h["reward"], h["cost_after"])
        for h in step_history
    )
    total_color = "#16a34a" if total >= 0 else "#dc2626"

    return (
        f'<div class="panel">'
        f'<p class="panel-title">⚡ Actions Timeline</p>'
        f'{entries_html}'
        f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;'
        f'display:flex;justify-content:space-between;font-size:0.83rem;">'
        f'<span style="color:#6b7280;">{len(step_history)} action(s)</span>'
        f'<span style="font-weight:700;color:{total_color};">Total: {sign}{total:.3f}</span>'
        f'</div>'
        f'</div>'
    )


def _bar(value: float, color: str = "#6366f1") -> str:
    pct = max(0, min(100, value * 100))
    return (
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>'
        f'</div>'
    )


def _score_color(v: float) -> str:
    if v >= 0.85: return "#22c55e"
    if v >= 0.65: return "#f59e0b"
    return "#ef4444"


def _score_icon(v: float) -> str:
    if v >= 0.85: return "🟢"
    if v >= 0.65: return "🟡"
    return "🔴"


def render_score(detail: dict[str, float], done: bool) -> str:
    if not done or not detail:
        return (
            '<div class="panel">'
            '<p class="panel-title">📊 Episode Score</p>'
            '<p style="color:#9ca3af;font-size:0.875rem;">Score will appear after the episode completes.</p>'
            '</div>'
        )

    total = detail.get("total_score", 0.0)
    total_icon = _score_icon(total)
    total_color = _score_color(total)

    bars_html = ""
    bar_colors = {
        "cost_reduction":       "#6366f1",
        "safety_maintained":    "#22c55e",
        "sequence_correctness": "#f59e0b",
        "no_risky_actions":     "#3b82f6",
        "efficiency":           "#8b5cf6",
    }

    for key, (label, weight) in SCORE_WEIGHTS.items():
        val = detail.get(key, 0.0)
        color = bar_colors.get(key, "#6366f1")
        bars_html += (
            f'<div class="score-row">'
            f'<div class="score-label">'
            f'<span style="color:#111827;">{label}</span>'
            f'<span style="font-family:monospace;font-weight:600;color:#111827;">{val:.2f} '
            f'<span style="color:#9ca3af;font-weight:400;">× {weight:.2f}</span></span>'
            f'</div>'
            f'{_bar(val, color)}'
            f'</div>'
        )

    return (
        f'<div class="panel">'
        f'<p class="panel-title">📊 Episode Score</p>'
        f'<div class="score-header">'
        f'<span style="color:{total_color};">{total:.3f}</span>'
        f'<span style="font-size:1.4rem;">{total_icon}</span>'
        f'<span style="font-size:0.9rem;font-weight:400;color:#6b7280;margin-left:4px;">'
        f'/ 1.000</span>'
        f'</div>'
        f'{bars_html}'
        f'</div>'
    )


def render_status(state: dict, last_action: str = "", last_reward: float = 0.0) -> str:
    if state["env"] is None:
        return '<div class="status-banner status-ready">Select a scenario and click Reset to begin.</div>'

    if not state["step_history"]:
        scenario = state.get("scenario", "easy")
        _, recommended = get_reasoning(state["obs"])
        return (
            f'<div class="status-banner status-ready">'
            f'✅ Environment ready — <strong>{scenario}</strong> scenario loaded. '
            f'Agent recommends: <strong>{recommended}</strong>'
            f'</div>'
        )

    if state["done"]:
        solved = state.get("grader_score", 0.0) >= 0.7
        cls = "status-done-ok" if solved else "status-done-to"
        icon = "🎯" if solved else "⏱️"
        outcome = "Episode solved!" if solved else "Episode timed out."
        score = state.get("grader_score", 0.0)
        return (
            f'<div class="status-banner {cls}">'
            f'{icon} {outcome} Final score: <strong>{score:.3f}</strong> — '
            f'Last action: <strong>{last_action}</strong> → '
            f'reward {"+" if last_reward >= 0 else ""}{last_reward:.3f}'
            f'</div>'
        )

    step = len(state["step_history"])
    _, next_action = get_reasoning(state["obs"])
    sign = "+" if last_reward >= 0 else ""
    return (
        f'<div class="status-banner status-running">'
        f'⚡ Step {step} complete — action: <strong>{last_action}</strong> → '
        f'reward <strong>{sign}{last_reward:.3f}</strong>. '
        f'Next recommended: <strong>{next_action}</strong>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _pack(state: dict, last_action: str = "", last_reward: float = 0.0) -> tuple:
    obs = state.get("obs")
    hist = state.get("step_history", [])
    done = state.get("done", False)
    detail = state.get("score_detail", {})

    return (
        state,
        render_scenario_card(state.get("scenario", "easy")),
        render_infra_state(obs),
        render_reasoning(hist),
        render_actions(hist),
        render_score(detail, done),
        render_status(state, last_action, last_reward),
    )


def _do_reset(scenario: str) -> tuple:
    state = empty_state()
    state["scenario"] = scenario

    env = CloudOpsEnv(scenario=scenario)
    obs = env.reset()

    state["env"] = env
    state["obs"] = obs
    state["initial_state"] = env.state()

    return _pack(state)


def _do_step(state: dict) -> tuple:
    if state["env"] is None:
        return _pack(state)

    if state["done"]:
        last = state["step_history"][-1] if state["step_history"] else {}
        return _pack(state, last.get("action", ""), last.get("reward", 0.0))

    obs: Observation = state["obs"]
    env: CloudOpsEnv = state["env"]

    reasoning_text, action_str = get_reasoning(obs)
    new_obs, reward, done, info = env.step(Action(action_type=action_str))

    state["actions_taken"].append(InternalAction[action_str])
    step_num = len(state["step_history"]) + 1
    state["step_history"].append({
        "step": step_num,
        "action": action_str,
        "reward": reward.value,
        "cost_after": new_obs.monthly_cost,
        "reasoning": reasoning_text,
    })
    state["obs"] = new_obs
    state["done"] = done

    if done:
        detail = grade_episode_detailed(
            state["actions_taken"],
            env.state(),
            state["initial_state"],
        )
        state["grader_score"] = detail["total_score"]
        state["score_detail"] = detail

    return _pack(state, action_str, reward.value)


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def reset_env(scenario: str, state: dict):
    return _do_reset(scenario)


def step_once(state: dict):
    return _do_step(state)


def run_episode(scenario: str, _state: dict):
    state_out = _do_reset(scenario)
    state = state_out[0]

    for _ in range(state["env"].max_steps):
        if state["done"]:
            break
        state_out = _do_step(state)
        state = state_out[0]

    return state_out


# ---------------------------------------------------------------------------
# OpenEnv REST API
# ---------------------------------------------------------------------------

class _ResetRequest(_BaseModel):
    task_name: str = "easy"


class _StepRequest(_BaseModel):
    action: dict


api = FastAPI(title="CloudOps Decision Gym API")
_api_env: CloudOpsEnv | None = None


@api.post("/reset")
async def api_reset(req: _ResetRequest = None):  # noqa: B008
    global _api_env
    task = (req.task_name if req else "easy") or "easy"
    try:
        _api_env = CloudOpsEnv(scenario=task)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    obs = _api_env.reset()
    return {"observation": obs.model_dump()}


@api.post("/step")
async def api_step(req: _StepRequest):
    global _api_env
    if _api_env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first"})
    try:
        action = Action(action_type=req.action.get("action_type", "NOOP"))
        obs, reward, done, info = _api_env.step(action)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    # Filter info to JSON-serialisable primitives
    safe_info = {
        k: v for k, v in info.items()
        if isinstance(v, (str, int, float, bool, type(None), dict))
    }
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": safe_info,
    }


class _GradeRequest(_BaseModel):
    actions: list = []
    initial_observation: dict = {}
    final_observation: dict = {}
    task_name: str = "easy"


@api.post("/grade")
async def api_grade(req: _GradeRequest):
    from graders.ec2_grader import grade_episode_detailed
    try:
        result = grade_episode_detailed(
            req.actions,
            req.final_observation,
            req.initial_observation,
        )
        return {"score": result["total_score"], "breakdown": result}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

OUTPUTS_COUNT = 7  # state + 6 HTML/Markdown components

with gr.Blocks(
    title="☁️ CloudOps Decision Gym",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    session_state = gr.State(empty_state())

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-block">
      <h1>☁️ CloudOps Decision Gym</h1>
      <p>Train AI agents to make safe, cost-effective cloud infrastructure decisions</p>
    </div>
    """)

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row(elem_classes="controls-row"):
        scenario_dd = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Scenario",
            scale=1,
            min_width=160,
        )
        reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1, min_width=120)
        step_btn  = gr.Button("⏭ Step", variant="primary",   scale=1, min_width=120)
        run_btn   = gr.Button("▶ Run AI Agent", variant="primary", scale=2, min_width=180)

    # ── Status banner ─────────────────────────────────────────────────────────
    status_html = gr.HTML(
        '<div class="status-banner status-ready">Select a scenario and click Reset to begin.</div>'
    )

    # ── Scenario card ─────────────────────────────────────────────────────────
    scenario_card = gr.HTML(render_scenario_card("easy"))

    # ── Control Room: 3 columns ───────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            infra_html = gr.HTML(render_infra_state(None))
        with gr.Column(scale=1):
            reasoning_html = gr.HTML(
                '<div class="panel"><p class="panel-title">🧠 Agent Reasoning</p>'
                '<p style="color:#9ca3af;font-size:0.875rem;">Reasoning will appear as the agent takes actions.</p>'
                '</div>'
            )
        with gr.Column(scale=1):
            actions_html = gr.HTML(
                '<div class="panel"><p class="panel-title">⚡ Actions Timeline</p>'
                '<p style="color:#9ca3af;font-size:0.875rem;">Actions will appear here.</p>'
                '</div>'
            )

    # ── Score panel ───────────────────────────────────────────────────────────
    score_html = gr.HTML(
        '<div class="panel"><p class="panel-title">📊 Episode Score</p>'
        '<p style="color:#9ca3af;font-size:0.875rem;">Score will appear after the episode completes.</p>'
        '</div>'
    )

    # ── Why this matters ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="why-section">
      <strong>⚠️ Why this matters:</strong>
      A wrong decision in a real cloud environment could mean
      <strong>unexpected downtime</strong> from an unsafe downsize,
      <strong>credential theft</strong> via an exposed IMDSv1 endpoint,
      or <strong>thousands of dollars in unnecessary spend</strong> from an over-provisioned instance.
      This environment trains AI agents to navigate those trade-offs safely —
      teaching them that <strong>security always comes before cost</strong>.
    </div>
    """)

    # ── Wiring ────────────────────────────────────────────────────────────────
    _outputs = [
        session_state,
        scenario_card,
        infra_html,
        reasoning_html,
        actions_html,
        score_html,
        status_html,
    ]

    reset_btn.click(fn=reset_env,    inputs=[scenario_dd, session_state], outputs=_outputs)
    step_btn.click( fn=step_once,    inputs=[session_state],               outputs=_outputs)
    run_btn.click(  fn=run_episode,  inputs=[scenario_dd, session_state], outputs=_outputs)
    scenario_dd.change(fn=reset_env, inputs=[scenario_dd, session_state], outputs=_outputs)


# ---------------------------------------------------------------------------
# ASGI app — Gradio mounted on FastAPI so /reset and /step also work
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(api, demo, path="/")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
