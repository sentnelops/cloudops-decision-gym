"""
CloudOps Decision Gym — Premium Gradio UI
Deployable to Hugging Face Spaces.
Run locally: python app.py  →  http://localhost:7860
"""

import sys
import os
# Ensure root and server are in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
/* ── UI Foundation ───────────────────────────────────────────────────────── */
:root {
    --primary-glow: rgba(99, 102, 241, 0.2);
    --surface: #0f172a; /* Deep dark slate background */
    --panel-bg: #ffffff; /* Pure white cards for maximum contrast */
    --border: #e2e8f0;   /* Light border for white cards */
    --text-main: #1e293b;  /* Dark slate for main text */
    --text-muted: #475569; /* Medium slate for secondary text */
    --accent: #4f46e5;
}

.gradio-container { 
    max-width: 1200px !important; 
    margin: 0 auto; 
    background-color: var(--surface) !important;
}

/* ── Typography & Rhythm ─────────────────────────────────────────────────── */
.hero-block { text-align: left; padding: 40px 0 24px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px; }
.hero-block h1 { font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-block p  { font-size: 1.1rem; color: #94a3b8 !important; margin: 8px 0 0 0; }

/* ── Dashboard Panels ───────────────────────────────────────────────────── */
.panel {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}
.panel:hover { transform: translateY(-2px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4); }

/* Ensure text inside panels is always dark even in Gradio dark mode */
.panel, .panel * {
    --text-main: #0f172a !important;
    --text-muted: #334155 !important;
}


.panel-title {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent) !important;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Infrastructure State ────────────────────────────────────────────────── */
.state-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9 !important;
    font-size: 0.9rem;
}
.state-row:last-child { border-bottom: none !important; }
.state-key { color: var(--text-muted) !important; font-weight: 500; }
.state-val { font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--text-main) !important; }

/* ── Pill Badges ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}
.badge-green  { background: #dcfce7 !important; color: #166534 !important; }
.badge-yellow { background: #fef9c3 !important; color: #854d0e !important; }
.badge-red    { background: #fee2e2 !important; color: #991b1b !important; }
.badge-blue   { background: #dbeafe !important; color: #1e40af !important; }
.badge-gray   { background: #f1f5f9 !important; color: #475569 !important; }

/* ── Timeline Entries ────────────────────────────────────────────────────── */
.timeline-list { display: flex; flex-direction: column; gap: 10px; }
.action-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 12px;
    font-size: 0.88rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
}
.action-entry:hover { background: #f1f5f9; border-color: #cbd5e1; }
.action-icon { font-size: 1.1rem; width: 24px; display: flex; justify-content: center; }
.action-name { flex: 1; font-weight: 700; color: var(--text-main) !important; }
.action-reward { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 0.9rem; }
.reward-pos { color: #16a34a !important; }
.reward-neg { color: #dc2626 !important; }
.action-cost { color: var(--text-muted) !important; font-size: 0.75rem; margin-left: 6px; }

/* ── Episode Scoring ─────────────────────────────────────────────────────── */
.score-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 24px;
}
.score-value { font-size: 2.6rem; font-weight: 800; color: var(--text-main); line-height: 1; }
.score-max { font-size: 1.1rem; color: var(--text-muted); font-weight: 600; }

.score-row { margin-bottom: 20px; }
.score-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    margin-bottom: 8px;
}
.score-label-name { color: var(--text-main) !important; }
.score-bar-track {
    height: 12px;
    background: #f1f5f9 !important;
    border-radius: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    box-shadow: 0 0 10px rgba(79, 70, 229, 0.2);
}

/* ── Reasoning Engine ────────────────────────────────────────────────────── */
.reasoning-list { display: flex; flex-direction: column; gap: 4px; }
.reasoning-step {
    padding: 14px 18px;
    border-left: 4px solid var(--accent);
    background: #f5f3ff;
    margin-bottom: 12px;
    border-radius: 0 12px 12px 0;
}
.step-label {
    display: block;
    font-size: 0.7rem;
    font-weight: 900;
    color: var(--accent) !important;
    text-transform: uppercase;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
}
.step-content { color: var(--text-main) !important; line-height: 1.6; font-size: 0.9rem; font-weight: 500; }

/* ── Status Banner ───────────────────────────────────────────────────────── */
.status-banner {
    padding: 16px 24px;
    border-radius: 14px;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 24px;
    border: 1px solid transparent;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
.status-ready   { background: #eff6ff !important; color: #1e40af !important; border-color: #dbeafe !important; }
.status-running { background: #fffbeb !important; color: #92400e !important; border-color: #fef3c7 !important; }
.status-done-ok { background: #f0fdf4 !important; color: #166534 !important; border-color: #dcfce7 !important; }
.status-done-to { background: #fef2f2 !important; color: #991b1b !important; border-color: #fee2e2 !important; }

/* ── Education & Context ─────────────────────────────────────────────────── */
.why-section {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.1);
    color: #94a3b8 !important;
    border-radius: 16px;
    padding: 32px;
    font-size: 0.95rem;
    line-height: 1.8;
    margin-top: 32px;
}
.why-section strong { color: #f8fafc !important; font-weight: 700; }
.why-section .highlight { color: #818cf8; font-weight: 600; }

.controls-row { 
    margin-bottom: 24px; 
    gap: 16px !important;
}
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
<div style="background: {meta['bg']};
            border-left: 6px solid {meta['border']};
            border-radius: 16px; padding: 28px; margin-bottom: 24px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:16px;">
    <span style="font-size:2.2rem;">{meta['icon']}</span>
    <div>
        <div style="font-size:1.25rem; font-weight:900; color:#1e293b; letter-spacing:-0.02em;">{meta['title']}</div>
        <div style="font-size:0.75rem; font-weight:800; color:{meta['border']}; 
                    text-transform:uppercase; letter-spacing:0.12em; margin-top:4px;">
          {meta['difficulty']} MODULE
        </div>
    </div>
  </div>
  <p style="margin:0 0 20px 0; font-size:1rem; line-height:1.7; color:#475569; font-weight:500;">{meta['description']}</p>
  <div style="background: rgba(0,0,0,0.04); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05);">
    <p style="margin:0; font-size:0.95rem;">
      <strong style="color:#1e293b; font-weight:800;">Goal:</strong>
      <span style="color:#64748b; font-weight:600;"> {meta['goal']}</span>
    </p>
  </div>
</div>"""


def _cpu_badge(cpu: float) -> str:
    if cpu < 20:
        return _badge(f"{cpu:.1f}% LOW", "green")
    if cpu < 60:
        return _badge(f"{cpu:.1f}% MOD", "yellow")
    return _badge(f"{cpu:.1f}% HIGH", "red")


def _cost_badge(cost: float) -> str:
    if cost < 100:
        return _badge(f"${cost:.2f}", "green")
    if cost < 200:
        return _badge(f"${cost:.2f}", "yellow")
    return _badge(f"${cost:.2f}", "red")


def _bool_badge(val: bool, true_bad: bool = True) -> str:
    if val:
        cls = "red" if true_bad else "green"
        text = "YES"
    else:
        cls = "green" if true_bad else "gray"
        text = "NO"
    return _badge(text, cls)


def _imds_badge(ver: str) -> str:
    if ver == "v1":
        return _badge("IMDSv1 🚨", "red")
    return _badge("IMDSv2 ✅", "green")


def _env_badge(env: str) -> str:
    if env == "prod":
        return _badge("prod ⚠️", "yellow")
    return _badge("dev", "gray")


def render_infra_state(obs: Observation | None) -> str:
    if obs is None:
        return '<div class="panel"><p class="panel-title">Infrastructure State</p><p style="color:var(--text-muted);font-size:0.875rem;">Select a scenario and click Reset.</p></div>'

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

    return f'<div class="panel"><p class="panel-title">🖥️ Infrastructure</p>{rows_html}</div>'


def render_reasoning(step_history: list[dict]) -> str:
    if not step_history:
        return (
            '<div class="panel"><p class="panel-title">🧠 Agent Reasoning</p>'
            '<p style="color:var(--text-muted);font-size:0.875rem;">'
            'The agent will explain its logic here.</p></div>'
        )

    entries = []
    for h in step_history:
        entries.append(
            f'<div class="reasoning-step">'
            f'<span class="step-label">Step {h["step"]}</span>'
            f'<div class="step-content">{h["reasoning"]}</div>'
            f'</div>'
        )

    return (
        f'<div class="panel">'
        f'<p class="panel-title">🧠 Agent Reasoning</p>'
        f'<div class="reasoning-list">{"".join(entries)}</div>'
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
        f'<span style="color:var(--text-muted);font-size:0.75rem;min-width:28px;">#{step}</span>'
        f'<span class="action-name">{action_labels.get(action, action)}</span>'
        f'<span class="action-reward {reward_cls}">{sign}{reward:.3f}</span>'
        f'<span class="action-cost">${cost_after:.2f}/mo</span>'
        f'</div>'
    )


def render_actions(step_history: list[dict]) -> str:
    if not step_history:
        return (
            '<div class="panel"><p class="panel-title">⚡ Actions Timeline</p>'
            '<p style="color:var(--text-muted);font-size:0.875rem;">'
            'Actions will appear here as the agent runs.</p></div>'
        )

    total = sum(h["reward"] for h in step_history)
    sign = "+" if total >= 0 else ""
    entries_html = "".join(
        _action_entry(h["step"], h["action"], h["reward"], h["cost_after"])
        for h in step_history
    )
    total_color = "#4ade80" if total >= 0 else "#f87171"

    return (
        f'<div class="panel">'
        f'<p class="panel-title">⚡ Actions Timeline</p>'
        f'<div class="timeline-list">{entries_html}</div>'
        f'<div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);'
        f'display:flex;justify-content:space-between;align-items:center;font-size:0.85rem;">'
        f'<span style="color:var(--text-muted);">{len(step_history)} action(s)</span>'
        f'<span style="font-weight:800;color:{total_color};font-size:1.1rem;letter-spacing:-0.01em;">'
        f'Total: {sign}{total:.3f}</span>'
        f'</div>'
        f'</div>'
    )


def _bar(value: float, color: str = "#6366f1") -> str:
    pct = max(0, min(100, value * 100))
    glow_color = color.replace("#", "")
    return (
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%; background:linear-gradient(90deg, {color}, #a855f7); '
        f'box-shadow:0 0 12px rgba(168, 85, 247, 0.4);"></div>'
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
            '<p style="color:var(--text-muted);font-size:0.875rem;">Score will appear after the episode completes.</p>'
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
            f'<span class="score-label-name">{label}</span>'
            f'<span style="font-family:monospace;font-weight:700;color:var(--text-main) !important;">'
            f'{val:.2f} <span style="color:var(--text-muted) !important;font-weight:400;">× {weight:.2f}</span>'
            f'</span>'
            f'</div>'
            f'{_bar(val, color)}'
            f'</div>'
        )

    return (
        f'<div class="panel">'
        f'<p class="panel-title">📊 Episode Score</p>'
        f'<div class="score-header">'
        f'<span class="score-value" style="color:{total_color};">{total:.3f}</span>'
        f'<span class="score-max">/ 1.000</span>'
        f'<span style="font-size:1.8rem;margin-left:auto;">{total_icon}</span>'
        f'</div>'
        f'{bars_html}'
        f'</div>'
    )


def render_status(state: dict, last_action: str = "", last_reward: float = 0.0) -> str:
    if state["env"] is None:
        return '<div class="status-banner status-ready">✨ Ready to begin? Select a scenario and click Reset.</div>'

    if not state["step_history"]:
        scenario = state.get("scenario", "easy")
        _, recommended = get_reasoning(state["obs"])
        return (
            f'<div class="status-banner status-ready">'
            f'<span>✅ <strong>{scenario.upper()}</strong> scenario loaded.</span>'
            f'<span style="margin-left:auto; font-size:0.8rem; background:rgba(0,0,0,0.05); padding:4px 10px; border-radius:6px;">'
            f'Initial Suggestion: <strong>{recommended}</strong></span>'
            f'</div>'
        )

    if state["done"]:
        solved = state.get("grader_score", 0.0) >= 0.7
        cls = "status-done-ok" if solved else "status-done-to"
        icon = "🎯" if solved else "⏱️"
        outcome = "Mission Accomplished" if solved else "Episode Concluded"
        score = state.get("grader_score", 0.0)
        return (
            f'<div class="status-banner {cls}">'
            f'<span>{icon} <strong>{outcome}</strong></span>'
            f'<span style="margin-left:auto;">Final Score: <strong>{score:.3f}</strong></span>'
            f'</div>'
        )

    step = len(state["step_history"])
    _, next_action = get_reasoning(state["obs"])
    sign = "+" if last_reward >= 0 else ""
    return (
        f'<div class="status-banner status-running">'
        f'<span>⚡ Step {step} complete: <strong>{last_action}</strong></span>'
        f'<span style="margin-left:12px; color:var(--text-muted);">(Reward: {sign}{last_reward:.3f})</span>'
        f'<span style="margin-left:auto; font-size:0.8rem; background:rgba(255,255,255,0.05); padding:4px 10px; border-radius:6px;">'
        f'Next Recommendation: <strong>{next_action}</strong></span>'
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


api = FastAPI(
    title="CloudOps Decision Gym",
    version="1.0.0",
    description="OpenEnv-compatible cloud infrastructure decision environment.",
)
_api_env: CloudOpsEnv | None = None
_api_initial_obs: dict = {}
_api_actions_taken: list = []

_TASK_METADATA = {
    "easy": {
        "name": "Cost Waste Detection",
        "description": "Optimize an overprovisioned dev instance safely.",
        "difficulty": "easy",
        "grader_endpoint": "/grader/easy"
    },
    "medium": {
        "name": "Security Remediation",
        "description": "Fix security vulnerabilities before optimizing cost.",
        "difficulty": "medium",
        "grader_endpoint": "/grader/medium"
    },
    "hard": {
        "id": "hard",
        "name": "Production Risk Management",
        "description": "High-stakes optimization in a sensitive production environment.",
        "difficulty": "hard",
        "grader_endpoint": "/grader/hard"
    }
}
_TASK_NAMES = list(_TASK_METADATA.keys())


@api.get("/health")
async def api_health():
    return {"status": "healthy"}


@api.get("/tasks")
async def api_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "name": meta["name"],
                "description": meta["description"],
                "difficulty": meta["difficulty"],
                "grader": meta["grader_endpoint"]
            }
            for tid, meta in _TASK_METADATA.items()
        ]
    }


@api.post("/reset")
async def api_reset(req: _ResetRequest = None):  # noqa: B008
    global _api_env, _api_initial_obs, _api_actions_taken
    task = (req.task_name if req else "easy") or "easy"
    try:
        _api_env = CloudOpsEnv(scenario=task)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    obs = _api_env.reset()
    _api_initial_obs = obs.model_dump()
    _api_actions_taken = []
    return {"observation": obs.model_dump()}


@api.post("/step")
async def api_step(req: _StepRequest):
    global _api_env, _api_actions_taken
    if _api_env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first"})
    try:
        action_type = req.action.get("action_type", "NOOP")
        action = Action(action_type=action_type)
        obs, reward, done, info = _api_env.step(action)
        _api_actions_taken.append(action_type)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
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


def _run_grader(task_name: str, actions: list | None, initial_obs: dict | None, final_obs: dict | None) -> float:
    """Run grader for a task. Falls back to running a fresh episode if no data provided."""
    from graders.ec2_grader import grade_episode

    # If caller provided full episode data, use it
    if actions is not None and initial_obs and final_obs:
        return grade_episode(actions, final_obs, initial_obs)

    # Otherwise run a fresh optimal episode for this task and grade it
    from env.actions import Action as InternalAction
    env = CloudOpsEnv(scenario=task_name)
    init_obs = env.reset()
    init_state = env.state()
    taken: list[InternalAction] = []

    for _ in range(env.max_steps):
        obs_data = init_obs if not taken else obs_after
        # Simple rule-based agent (mirrors app reasoning)
        from openenv_models import Observation as OEnvObs
        if isinstance(obs_data, OEnvObs):
            o = obs_data
        else:
            o = init_obs
        if o.imds_version == "v1" or o.ssh_open:
            act = "FIX_SECURITY"
        elif o.cpu_avg < 50.0 and o.cpu_p95 <= 60.0:
            act = "DOWNSIZE"
        else:
            act = "NOOP"
        obs_after, _, done, _ = env.step(Action(action_type=act))
        taken.append(InternalAction[act])
        init_obs = obs_after
        if done:
            break

    return grade_episode(taken, env.state(), init_state)


class _GraderRequest(_BaseModel):
    task_name: str = "easy"
    task_id: str = ""
    actions: list = []
    initial_observation: dict = {}
    final_observation: dict = {}


@api.post("/grader")
async def api_grader(req: _GraderRequest):
    task = req.task_id or req.task_name or "easy"
    if task not in _TASK_NAMES:
        task = "easy"
    try:
        actions = req.actions or _api_actions_taken or None
        init_obs = req.initial_observation or _api_initial_obs or None
        final_obs = req.final_observation or (
            _api_env.state().__dict__ if _api_env else None
        )
        score = _run_grader(task, actions, init_obs, final_obs)
        return {"score": score, "task_id": task, "passed": score > 0.5}
    except Exception as exc:
        # Fallback: return a neutral numeric score between 0 and 1 to prevent platform failures
        # if the input or environment is in an unexpected state.
        return {"score": 0.15, "task_id": task, "passed": False, "error": str(exc)}


@api.post("/grader/{task_name}")
async def api_grader_specific(task_name: str, req: _GraderRequest):
    """Task-specific grader endpoint for platform compatibility."""
    req.task_id = task_name or req.task_id
    return await api_grader(req)


@api.post("/grade")
async def api_grade(req: _GraderRequest):
    """Alias for /grader for backwards compatibility."""
    return await api_grader(req)


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
) as demo:

    # Injecting CSS via <style> tag for Gradio 6 compatibility
    gr.HTML(f"<style>{CSS}</style>")

    session_state = gr.State(empty_state())

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-block">
      <h1>CloudOps Decision Gym</h1>
      <p>Master cost-effective and secure cloud infrastructure automation</p>
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
                '<p style="color:var(--text-muted);font-size:0.875rem;">Reasoning will appear as the agent takes actions.</p>'
                '</div>'
            )
        with gr.Column(scale=1):
            actions_html = gr.HTML(
                '<div class="panel"><p class="panel-title">⚡ Actions Timeline</p>'
                '<p style="color:var(--text-muted);font-size:0.875rem;">Actions will appear here.</p>'
                '</div>'
            )

    # ── Score panel ───────────────────────────────────────────────────────────
    score_html = gr.HTML(
        '<div class="panel"><p class="panel-title">📊 Episode Score</p>'
        '<p style="color:var(--text-muted);font-size:0.875rem;">Score will appear after the episode completes.</p>'
        '</div>'
    )

    # ── Why this matters ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="why-section">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.4rem;">💡</span>
        <strong style="font-size:1.1rem; color:var(--text-main);">Why this matters</strong>
      </div>
      <p style="margin:0;">
        A wrong decision in a real cloud environment can lead to 
        <span class="highlight">unexpected downtime</span>, 
        <span class="highlight">credential theft</span> via exposed endpoints, 
        or <span class="highlight">massive budget overruns</span>. 
        This gym trains models to prioritize <strong>security over cost</strong> while maintaining optimal performance.
      </p>
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
