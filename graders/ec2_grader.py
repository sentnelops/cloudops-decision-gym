"""
CloudOps Decision Gym — EC2 Episode Grader

Deterministic, post-episode scoring.
Evaluates the quality of an agent's decisions across a full episode.
Returns a score in [0.0, 1.0].

Phase 2 updates:
    - Added sequence_correctness criterion (0.25 weight)
    - Rebalanced weights to reflect security-first philosophy
    - grade_episode_detailed() includes sequence_correctness key

Scoring criteria (weights sum to 1.0):
    cost_reduction       (0.25)  Did cost meaningfully decrease?
    safety_maintained    (0.25)  No security regressions?
    sequence_correctness (0.25)  Did security get fixed before downsizing?
    no_risky_actions     (0.15)  No dangerous downsizes under high load?
    efficiency           (0.10)  No wasteful NOOPs or redundant steps?
"""

import os as _os
import sys as _sys

# Ensure the package root is on sys.path so this module is importable
# from any working directory (e.g. when the OpenEnv platform loads it).
_GRADERS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PKG_ROOT = _os.path.dirname(_GRADERS_DIR)
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from constants import (
    CPU_P95_DOWNSIZE_LIMIT,
    MEMORY_DOWNSIZE_LIMIT,
)
from env.actions import Action
from models import EC2State

# ---------------------------------------------------------------------------
# Input normalisation — accept strings/dicts/Pydantic models from the platform
# ---------------------------------------------------------------------------

def _normalise_actions(actions: list) -> list[Action]:
    """Convert any action representation to internal Action IntEnum."""
    result = []
    for a in actions:
        if isinstance(a, Action):
            result.append(a)
        elif isinstance(a, str):
            result.append(Action[a.upper()])
        elif isinstance(a, dict):
            key = a.get("action_type") or a.get("action") or "NOOP"
            result.append(Action[key.upper()])
        else:
            result.append(Action(int(a)))
    return result


def _normalise_state(state) -> EC2State:
    """Convert Observation, plain dict, or EC2State to EC2State."""
    if isinstance(state, EC2State):
        return state
    d = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    return EC2State(
        instance_type=d["instance_type"],
        cpu_avg=float(d["cpu_avg"]),
        cpu_p95=float(d["cpu_p95"]),
        memory_avg=float(d["memory_avg"]),
        monthly_cost=float(d["monthly_cost"]),
        internet_facing=bool(d["internet_facing"]),
        ssh_open=bool(d["ssh_open"]),
        imds_version=d["imds_version"],
        environment=d["environment"],
    )


# Grading weights — must sum to 1.0
_W_COST = 0.25
_W_SAFETY = 0.25
_W_SEQUENCE = 0.25
_W_RISK = 0.15
_W_EFFICIENCY = 0.10


def grade_episode(
    actions: list,
    final_state,
    initial_state,
) -> float:
    """
    Score a completed episode based on final outcomes and action history.

    Accepts actions as Action IntEnum, strings, or dicts.
    Accepts states as EC2State, Observation (Pydantic), or plain dicts.

    Returns:
        float strictly in (0.0, 1.0) — higher is better.
    """
    actions = _normalise_actions(actions)
    final_state = _normalise_state(final_state)
    initial_state = _normalise_state(initial_state)

    cost_score = _score_cost_reduction(initial_state, final_state)
    safety_score = _score_safety_maintained(initial_state, final_state)
    sequence_score = _score_sequence_correctness(actions, initial_state)
    risk_score = _score_no_risky_actions(actions, initial_state)
    efficiency_score = _score_efficiency(actions)

    total = (
        _W_COST * cost_score
        + _W_SAFETY * safety_score
        + _W_SEQUENCE * sequence_score
        + _W_RISK * risk_score
        + _W_EFFICIENCY * efficiency_score
    )
    # Clamp to strictly (0, 1) and ensure it's not exactly 0.0 or 1.0
    # Range [0.051, 0.949] is safe and clearly within the required interval.
    return round(float(max(0.051, min(0.949, total))), 4)


def grade_episode_detailed(
    actions: list,
    final_state,
    initial_state,
) -> dict[str, float]:
    """
    Same as grade_episode but returns per-criterion scores for diagnostics.

    Accepts actions as Action IntEnum, strings, or dicts.
    Accepts states as EC2State, Observation (Pydantic), or plain dicts.

    Returns:
        dict with keys: cost_reduction, safety_maintained, sequence_correctness,
                        no_risky_actions, efficiency, total_score
    """
    actions = _normalise_actions(actions)
    final_state = _normalise_state(final_state)
    initial_state = _normalise_state(initial_state)

    cost_score = _score_cost_reduction(initial_state, final_state)
    safety_score = _score_safety_maintained(initial_state, final_state)
    sequence_score = _score_sequence_correctness(actions, initial_state)
    risk_score = _score_no_risky_actions(actions, initial_state)
    efficiency_score = _score_efficiency(actions)

    total = (
        _W_COST * cost_score
        + _W_SAFETY * safety_score
        + _W_SEQUENCE * sequence_score
        + _W_RISK * risk_score
        + _W_EFFICIENCY * efficiency_score
    )
    return {
        "cost_reduction": round(cost_score, 4),
        "safety_maintained": round(safety_score, 4),
        "sequence_correctness": round(sequence_score, 4),
        "no_risky_actions": round(risk_score, 4),
        "efficiency": round(efficiency_score, 4),
        "total_score": round(float(max(0.051, min(0.949, total))), 4),
    }


# ---------------------------------------------------------------------------
# Criterion functions
# ---------------------------------------------------------------------------

def _score_cost_reduction(initial: EC2State, final: EC2State) -> float:
    """
    Score = (initial_cost - final_cost) / initial_cost, clamped to [0.0, 1.0].
    A 40% cost reduction → 0.40 score; 100% → 1.0.
    No reduction or cost increase → 0.0.
    """
    if initial.monthly_cost == 0.0:
        return 1.0 if final.monthly_cost == 0.0 else 0.0
    reduction_ratio = (initial.monthly_cost - final.monthly_cost) / initial.monthly_cost
    return max(0.0, min(1.0, reduction_ratio))


def _score_safety_maintained(initial: EC2State, final: EC2State) -> float:
    """
    Full score (1.0) if no security regressions occurred.
    0.0 if any security property is worse at the end than at the start.

    Regressions checked:
        - imds_version moved from "v2" back to "v1"
        - ssh_open went from False → True
        - internet_facing went from False → True

    Note: this criterion rewards not making things WORSE. Improvement is
    captured by sequence_correctness.
    """
    if initial.imds_version == "v2" and final.imds_version == "v1":
        return 0.0
    if not initial.ssh_open and final.ssh_open:
        return 0.0
    if not initial.internet_facing and final.internet_facing:
        return 0.0
    return 1.0


def _score_sequence_correctness(
    actions: list[Action], initial_state: EC2State
) -> float:
    """
    Evaluates whether the agent respected the security-first ordering principle.

    If the initial state had NO security issues: sequencing is irrelevant → 1.0
    If the initial state HAD security issues:
        - FIX_SECURITY before any DOWNSIZE                         → 1.0  (perfect)
        - FIX_SECURITY after some DOWNSIZEs (eventually fixed)     → 0.5  (partial)
        - No FIX_SECURITY, but DOWNSIZEs occurred                  → 0.0  (ignored security)
        - No DOWNSIZE at all (only fixed security, no cost action)  → 0.9  (safe, incomplete)
        - No actions taken at all                                   → 0.0

    The "safe but incomplete" case (score 0.9) reflects that security-only is better
    than wrong-sequence, but a full solution requires both fixes.
    """
    if not actions:
        return 0.0

    had_security_issues = not initial_state.is_secure

    # No initial security issues — ordering doesn't apply
    if not had_security_issues:
        return 1.0

    # Find ordering of first FIX_SECURITY and first DOWNSIZE
    first_fix_idx: int | None = None
    first_downsize_idx: int | None = None

    for i, action in enumerate(actions):
        if action == Action.FIX_SECURITY and first_fix_idx is None:
            first_fix_idx = i
        if action == Action.DOWNSIZE and first_downsize_idx is None:
            first_downsize_idx = i

    has_fix = first_fix_idx is not None
    has_downsize = first_downsize_idx is not None

    if has_fix and not has_downsize:
        # Security fixed but no downsize attempted — safe but incomplete
        return 0.9

    if has_fix and has_downsize:
        if first_fix_idx < first_downsize_idx:
            # Correct order: FIX_SECURITY came before DOWNSIZE
            return 1.0
        else:
            # Wrong order: DOWNSIZE happened before FIX_SECURITY
            return 0.5

    if not has_fix and has_downsize:
        # Security ignored entirely while downsizing
        return 0.0

    # No relevant actions (all NOOPs or UPSIZEs)
    return 0.0


def _score_no_risky_actions(actions: list[Action], initial_state: EC2State) -> float:
    """
    Penalizes DOWNSIZE actions taken when the initial state showed high load.

    Uses initial state as a proxy for conditions at decision time.
    If initial cpu_p95 or memory was above the downsize safety limit,
    any DOWNSIZE in the episode is considered risky.
    """
    downsize_count = actions.count(Action.DOWNSIZE)

    initial_high_p95 = initial_state.cpu_p95 > CPU_P95_DOWNSIZE_LIMIT
    initial_high_mem = initial_state.memory_avg > MEMORY_DOWNSIZE_LIMIT

    if downsize_count > 0 and (initial_high_p95 or initial_high_mem):
        return 0.0

    return 1.0


def _score_efficiency(actions: list[Action]) -> float:
    """
    Penalizes wasted or redundant actions.

    Deductions:
        - Each NOOP: -0.2 (capped at -0.6 total)

    Score floor: 0.0
    """
    if not actions:
        return 0.0

    noop_count = actions.count(Action.NOOP)
    noop_penalty = min(0.6, noop_count * 0.2)

    return max(0.0, 1.0 - noop_penalty)
