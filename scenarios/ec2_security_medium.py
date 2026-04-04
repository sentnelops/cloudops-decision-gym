"""
CloudOps Decision Gym — Scenario: EC2 Security Conflict (Medium)

Setup:
    An m5.xlarge dev instance with low CPU but critical security vulnerabilities:
    - IMDSv1 enabled (SSRF/credential theft risk)
    - SSH port 22 open to 0.0.0.0/0
    - Publicly internet-facing

    The instance is overprovisioned, so cost optimization is also valid —
    but ONLY after the security posture is hardened.

Problem:
    The agent must learn that security remediation has higher priority
    than cost optimization. Downsizing before fixing security is penalized.

Expected optimal sequence:
    Step 1: FIX_SECURITY  (remediate IMDSv1 + close SSH)
    Step 2: DOWNSIZE      (now safe to resize)
    Step 3: DOWNSIZE      (optional — further cost reduction)

Solved condition:
    instance.is_secure AND cost reduced by ≥ 20%
"""

from constants import INSTANCE_MONTHLY_COST
from env.actions import Action
from models import EC2State

_INITIAL_INSTANCE: str = "m5.xlarge"
_INITIAL_COST: float = INSTANCE_MONTHLY_COST[_INITIAL_INSTANCE]
_COST_REDUCTION_THRESHOLD: float = 0.20   # 20% reduction required


class EC2SecurityMediumScenario:
    """
    Medium-tier scenario: overprovisioned dev instance with active security issues.

    The agent must prioritize security over cost optimization.
    Attempting DOWNSIZE before FIX_SECURITY incurs a sequencing penalty.
    """

    description: str = (
        "A dev m5.xlarge instance at 12% CPU, publicly internet-facing with IMDSv1 "
        "enabled and SSH open to 0.0.0.0/0. Security must be remediated before "
        "cost optimization. Optimal: FIX_SECURITY → DOWNSIZE."
    )

    optimal_first_action: Action = Action.FIX_SECURITY

    max_steps: int = 7

    @property
    def initial_state(self) -> EC2State:
        """Fixed, deterministic initial state."""
        return EC2State(
            instance_type=_INITIAL_INSTANCE,
            cpu_avg=12.0,
            cpu_p95=18.0,
            memory_avg=22.0,
            monthly_cost=_INITIAL_COST,           # $175.20
            internet_facing=True,
            ssh_open=True,
            imds_version="v1",
            environment="dev",
        )

    def is_solved(self, state: EC2State) -> bool:
        """
        Solved when:
            1. Security posture is fully clean (IMDSv2 + SSH closed), AND
            2. Monthly cost reduced by at least 20% from initial.

        Both conditions must hold. An agent that only fixes security without
        downsizing does NOT solve the episode.
        """
        cost_reduction = (_INITIAL_COST - state.monthly_cost) / _INITIAL_COST
        return state.is_secure and cost_reduction >= _COST_REDUCTION_THRESHOLD
