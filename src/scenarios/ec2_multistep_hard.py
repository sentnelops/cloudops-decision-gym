"""
CloudOps Decision Gym — Scenario: Multi-Step Production (Hard)

Setup:
    An m5.2xlarge PRODUCTION instance at moderate CPU load (35% avg, 52% p95)
    with full security exposure:
    - IMDSv1 enabled
    - SSH open to 0.0.0.0/0
    - Internet-facing

    The instance is both vulnerable AND overprovisioned — but because this is
    production, the stakes are higher and all penalties are amplified by
    PROD_RISK_MULTIPLIER (1.5×).

Problem:
    The agent must learn:
    1. Security is the highest priority — fix before ANY sizing change
    2. Production raises the cost of mistakes — wrong sequence is penalized heavily
    3. After security fix, a careful downsize is appropriate (cpu_p95=52 < 60 threshold)

Expected optimal sequence:
    Step 1: FIX_SECURITY  (critical: IMDS upgrade + close SSH)
    Step 2: DOWNSIZE      (safe: cpu_p95=52% is under the 60% limit)
    Step 3: DOWNSIZE      (optional: further savings if still overprovisioned)

Why it's hard:
    - Wrong sequence (DOWNSIZE first) gets PROD_SEQUENCING_PENALTY (-0.60)
    - NOOP on a vulnerable prod instance gets security_ignore penalty × 1.5
    - Agent must balance urgency (security) vs. caution (prod risk)

Solved condition:
    instance.is_secure AND cost reduced by ≥ 20%
"""

from constants import INSTANCE_MONTHLY_COST
from env.actions import Action
from models import EC2State

_INITIAL_INSTANCE: str = "m5.2xlarge"
_INITIAL_COST: float = INSTANCE_MONTHLY_COST[_INITIAL_INSTANCE]
_COST_REDUCTION_THRESHOLD: float = 0.20   # 20% cost reduction required


class EC2MultistepHardScenario:
    """
    Hard-tier scenario: vulnerable production instance with moderate CPU load.

    Combines security remediation urgency with prod-environment risk amplification.
    Tests whether the agent has internalized the security-first principle under pressure.
    """

    description: str = (
        "A prod m5.2xlarge at 35% avg / 52% p95 CPU with IMDSv1, open SSH, and "
        "internet-facing exposure. Production penalties are 1.5× — wrong decisions "
        "are heavily penalized. Optimal: FIX_SECURITY → DOWNSIZE."
    )

    optimal_first_action: Action = Action.FIX_SECURITY

    max_steps: int = 8

    @property
    def initial_state(self) -> EC2State:
        """Fixed, deterministic initial state."""
        return EC2State(
            instance_type=_INITIAL_INSTANCE,
            cpu_avg=35.0,
            cpu_p95=52.0,
            memory_avg=45.0,
            monthly_cost=_INITIAL_COST,           # $276.48
            internet_facing=True,
            ssh_open=True,
            imds_version="v1",
            environment="prod",
        )

    def is_solved(self, state: EC2State) -> bool:
        """
        Solved when:
            1. Security posture is fully clean (IMDSv2 + SSH closed), AND
            2. Monthly cost reduced by at least 20% from initial ($276.48).

        Note: cpu_p95=52% is below the 60% DOWNSIZE threshold, so a single
        DOWNSIZE after security fix is safe and sufficient to trigger solved.
        """
        cost_reduction = (_INITIAL_COST - state.monthly_cost) / _INITIAL_COST
        return state.is_secure and cost_reduction >= _COST_REDUCTION_THRESHOLD
