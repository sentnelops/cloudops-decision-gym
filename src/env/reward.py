"""
CloudOps Decision Gym — Reward Engine
Computes a continuous, multi-objective reward signal for each environment step.

Phase 2 additions:
    + sequencing_penalty  via ConstraintEngine (security-first, prod amplification)
"""

from constants import (
    MAX_MONTHLY_COST,
    CPU_P95_DOWNSIZE_LIMIT,
    MEMORY_DOWNSIZE_LIMIT,
    CPU_OVERPROVISION_MAX,
    COST_SAVING_SCORE_MAX,
    COST_SAVING_SCORE_MIN,
    SECURITY_IMDS_UPGRADE,
    SECURITY_SSH_CLOSE,
    PERFORMANCE_RISK_P95_PENALTY,
    PERFORMANCE_RISK_MEM_PENALTY,
    UNSAFE_UPSIZE_PENALTY,
    NOOP_OBVIOUS_PENALTY,
    REWARD_MIN,
    REWARD_MAX,
)
from env.actions import Action
from env.constraints import ConstraintEngine
from models import EC2State


class RewardEngine:
    """
    Computes the scalar reward for a single (prev_state, action, next_state) transition.

    Reward components:
        + cost_saving_score           positive when cost reduced
        + security_improvement_score  positive when security posture improves
        - performance_risk_penalty    negative when action risks performance degradation
        - unsafe_action_penalty       negative for clearly wrong actions
        - sequencing_penalty          negative for violating security-first ordering
                                      (amplified in prod environments)

    Final reward is clamped to [REWARD_MIN, REWARD_MAX].
    """

    def __init__(self) -> None:
        self._constraints = ConstraintEngine()

    def compute(
        self,
        prev_state: EC2State,
        action: Action,
        next_state: EC2State,
    ) -> float:
        cost_component = self._cost_saving_score(prev_state, next_state)
        security_component = self._security_improvement_score(prev_state, next_state)
        perf_penalty = self._performance_risk_penalty(prev_state, action)
        unsafe_penalty = self._unsafe_action_penalty(action, prev_state)
        seq_penalty = self._sequencing_penalty(prev_state, action)

        raw = (
            cost_component
            + security_component
            + perf_penalty
            + unsafe_penalty
            + seq_penalty
        )
        return max(REWARD_MIN, min(REWARD_MAX, raw))

    def breakdown(
        self,
        prev_state: EC2State,
        action: Action,
        next_state: EC2State,
    ) -> dict[str, float]:
        """Returns each reward component for diagnostics/logging."""
        cost_component = self._cost_saving_score(prev_state, next_state)
        security_component = self._security_improvement_score(prev_state, next_state)
        perf_penalty = self._performance_risk_penalty(prev_state, action)
        unsafe_penalty = self._unsafe_action_penalty(action, prev_state)
        seq_penalty = self._sequencing_penalty(prev_state, action)

        raw = (
            cost_component
            + security_component
            + perf_penalty
            + unsafe_penalty
            + seq_penalty
        )
        final = max(REWARD_MIN, min(REWARD_MAX, raw))
        return {
            "cost_saving_score": round(cost_component, 4),
            "security_improvement_score": round(security_component, 4),
            "performance_risk_penalty": round(perf_penalty, 4),
            "unsafe_action_penalty": round(unsafe_penalty, 4),
            "sequencing_penalty": round(seq_penalty, 4),
            "raw_total": round(raw, 4),
            "final_reward": round(final, 4),
        }

    # ------------------------------------------------------------------
    # Component functions
    # ------------------------------------------------------------------

    def _cost_saving_score(self, prev: EC2State, next: EC2State) -> float:
        """
        Normalized cost delta.
        Range: [COST_SAVING_SCORE_MIN, COST_SAVING_SCORE_MAX]
        Positive = cost decreased, negative = cost increased.
        """
        if MAX_MONTHLY_COST == 0.0:
            return 0.0
        delta = prev.monthly_cost - next.monthly_cost  # positive = savings
        normalized = delta / MAX_MONTHLY_COST
        return max(COST_SAVING_SCORE_MIN, min(COST_SAVING_SCORE_MAX, normalized))

    def _security_improvement_score(self, prev: EC2State, next: EC2State) -> float:
        """
        Rewards improvements in security posture.
        Does NOT penalize lack of security action — that is handled elsewhere.
        """
        score = 0.0
        if prev.imds_version == "v1" and next.imds_version == "v2":
            score += SECURITY_IMDS_UPGRADE
        if prev.ssh_open and not next.ssh_open:
            score += SECURITY_SSH_CLOSE
        return score

    def _performance_risk_penalty(self, prev: EC2State, action: Action) -> float:
        """
        Penalizes DOWNSIZE when utilization is high enough to risk instability.
        Applied based on the state BEFORE the action (the risk was present at decision time).
        """
        if action != Action.DOWNSIZE:
            return 0.0
        penalty = 0.0
        if prev.cpu_p95 > CPU_P95_DOWNSIZE_LIMIT:
            penalty += PERFORMANCE_RISK_P95_PENALTY
        if prev.memory_avg > MEMORY_DOWNSIZE_LIMIT:
            penalty += PERFORMANCE_RISK_MEM_PENALTY
        return penalty

    def _unsafe_action_penalty(self, action: Action, state: EC2State) -> float:
        """
        Penalizes actions that are clearly wrong given the current state:
        - Upsizing an already overprovisioned instance
        - Doing nothing when the instance is obviously overprovisioned AND secure
          (security_ignore in ConstraintEngine handles the insecure NOOP case)
        """
        if action == Action.UPSIZE and state.cpu_avg < CPU_OVERPROVISION_MAX:
            return UNSAFE_UPSIZE_PENALTY
        if action == Action.NOOP and state.is_overprovisioned and state.is_secure:
            return NOOP_OBVIOUS_PENALTY
        return 0.0

    def _sequencing_penalty(self, state: EC2State, action: Action) -> float:
        """
        Sums penalties from all ConstraintEngine violations for this (state, action).
        Handles security-first ordering and prod environment amplification.
        """
        violations = self._constraints.evaluate(state, action)
        return sum(v.penalty for v in violations)
