"""
CloudOps Decision Gym — Action Space
Defines the discrete action set and applies transitions to EC2State.
"""

from dataclasses import dataclass
from enum import IntEnum

from constants import INSTANCE_HIERARCHY, INSTANCE_MONTHLY_COST
from models import EC2State


class Action(IntEnum):
    DOWNSIZE = 0      # Move to the next smaller instance type
    UPSIZE = 1        # Move to the next larger instance type
    FIX_SECURITY = 2  # Upgrade IMDSv1→v2 and close open SSH port 22
    NOOP = 3          # Take no action


@dataclass(frozen=True)
class ActionResult:
    """Outcome of attempting to apply an action."""
    applied: bool    # Whether the action produced a state change
    reason: str      # Human-readable explanation


class ActionSpace:
    """
    Validates and applies actions to EC2State instances.
    All methods are pure — no mutations, no side effects.
    """

    def is_valid(self, action: Action, state: EC2State) -> bool:
        """Returns True if the action can be meaningfully applied."""
        if action == Action.DOWNSIZE:
            idx = INSTANCE_HIERARCHY.index(state.instance_type)
            return idx > 0
        if action == Action.UPSIZE:
            idx = INSTANCE_HIERARCHY.index(state.instance_type)
            return idx < len(INSTANCE_HIERARCHY) - 1
        if action == Action.FIX_SECURITY:
            return state.imds_version == "v1" or state.ssh_open
        if action == Action.NOOP:
            return True
        return False

    def apply(self, action: Action, state: EC2State) -> tuple[EC2State, ActionResult]:
        """
        Apply action to state and return (new_state, result).
        If the action cannot be applied, the original state is returned unchanged.
        """
        if action == Action.DOWNSIZE:
            return self._apply_downsize(state)
        if action == Action.UPSIZE:
            return self._apply_upsize(state)
        if action == Action.FIX_SECURITY:
            return self._apply_fix_security(state)
        if action == Action.NOOP:
            return state, ActionResult(applied=False, reason="NOOP: no change")
        raise ValueError(f"Unknown action: {action!r}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_downsize(self, state: EC2State) -> tuple[EC2State, ActionResult]:
        idx = INSTANCE_HIERARCHY.index(state.instance_type)
        if idx == 0:
            return state, ActionResult(
                applied=False,
                reason=f"DOWNSIZE blocked: {state.instance_type} is the smallest instance type",
            )
        new_type = INSTANCE_HIERARCHY[idx - 1]
        new_cost = INSTANCE_MONTHLY_COST[new_type]
        new_state = EC2State(
            instance_type=new_type,
            cpu_avg=state.cpu_avg,
            cpu_p95=state.cpu_p95,
            memory_avg=state.memory_avg,
            monthly_cost=new_cost,
            internet_facing=state.internet_facing,
            ssh_open=state.ssh_open,
            imds_version=state.imds_version,
            environment=state.environment,
        )
        return new_state, ActionResult(
            applied=True,
            reason=f"DOWNSIZE: {state.instance_type} → {new_type} (${state.monthly_cost:.2f} → ${new_cost:.2f}/mo)",
        )

    def _apply_upsize(self, state: EC2State) -> tuple[EC2State, ActionResult]:
        idx = INSTANCE_HIERARCHY.index(state.instance_type)
        if idx == len(INSTANCE_HIERARCHY) - 1:
            return state, ActionResult(
                applied=False,
                reason=f"UPSIZE blocked: {state.instance_type} is the largest instance type",
            )
        new_type = INSTANCE_HIERARCHY[idx + 1]
        new_cost = INSTANCE_MONTHLY_COST[new_type]
        new_state = EC2State(
            instance_type=new_type,
            cpu_avg=state.cpu_avg,
            cpu_p95=state.cpu_p95,
            memory_avg=state.memory_avg,
            monthly_cost=new_cost,
            internet_facing=state.internet_facing,
            ssh_open=state.ssh_open,
            imds_version=state.imds_version,
            environment=state.environment,
        )
        return new_state, ActionResult(
            applied=True,
            reason=f"UPSIZE: {state.instance_type} → {new_type} (${state.monthly_cost:.2f} → ${new_cost:.2f}/mo)",
        )

    def _apply_fix_security(self, state: EC2State) -> tuple[EC2State, ActionResult]:
        if state.imds_version == "v2" and not state.ssh_open:
            return state, ActionResult(
                applied=False,
                reason="FIX_SECURITY: no issues to fix (IMDSv2 active, SSH closed)",
            )
        changes: list[str] = []
        new_imds = state.imds_version
        new_ssh = state.ssh_open
        if state.imds_version == "v1":
            new_imds = "v2"
            changes.append("IMDSv1 → IMDSv2")
        if state.ssh_open:
            new_ssh = False
            changes.append("SSH port 22 closed")
        new_state = EC2State(
            instance_type=state.instance_type,
            cpu_avg=state.cpu_avg,
            cpu_p95=state.cpu_p95,
            memory_avg=state.memory_avg,
            monthly_cost=state.monthly_cost,
            internet_facing=state.internet_facing,
            ssh_open=new_ssh,
            imds_version=new_imds,
            environment=state.environment,
        )
        return new_state, ActionResult(
            applied=True,
            reason=f"FIX_SECURITY: {', '.join(changes)}",
        )
