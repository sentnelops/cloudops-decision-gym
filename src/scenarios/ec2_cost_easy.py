"""
CloudOps Decision Gym — Scenario: EC2 Cost Optimization (Easy)

Setup:
    An m5.2xlarge dev instance running at 7% average CPU.
    It is overprovisioned by a factor of ~10x.
    Security posture is already clean (IMDSv2, SSH closed).

Optimal behavior:
    Agent should DOWNSIZE repeatedly until reaching a right-sized instance.
    A single DOWNSIZE is sufficient to score well on the easy tier.

Solved condition:
    Instance type is t3.large or smaller AND cost reduced by ≥40%.
"""

from constants import INSTANCE_MONTHLY_COST
from env.actions import Action
from models import EC2State

# Instances that satisfy the "solved" condition (t3.large or below in hierarchy)
_SOLVED_INSTANCE_TYPES: frozenset[str] = frozenset({
    "t3.micro",
    "t3.small",
    "t3.medium",
    "t3.large",
})

_COST_REDUCTION_THRESHOLD: float = 0.40   # must reduce cost by at least 40%


class EC2CostEasyScenario:
    """
    Easy-tier scenario: overprovisioned dev EC2 instance.

    The correct strategy is to DOWNSIZE until reaching a right-sized
    instance. No security remediation is needed (posture is clean).
    """

    description: str = (
        "A dev m5.2xlarge instance running at 7% average CPU with 18% memory usage. "
        "The instance is overprovisioned by ~10x. Security posture is clean. "
        "Optimal action: DOWNSIZE to a cost-appropriate instance."
    )

    optimal_first_action: Action = Action.DOWNSIZE

    max_steps: int = 5

    @property
    def initial_state(self) -> EC2State:
        """
        Fixed initial state — always the same (deterministic).
        Returns a new EC2State object each time to preserve immutability.
        """
        return EC2State(
            instance_type="m5.2xlarge",
            cpu_avg=7.0,
            cpu_p95=12.0,
            memory_avg=18.0,
            monthly_cost=INSTANCE_MONTHLY_COST["m5.2xlarge"],  # $276.48
            internet_facing=False,
            ssh_open=False,
            imds_version="v2",
            environment="dev",
        )

    def is_solved(self, state: EC2State) -> bool:
        """
        Episode is solved when:
            1. Instance type is t3.large or smaller, AND
            2. Monthly cost is at least 40% lower than the initial cost.

        Both conditions must hold simultaneously.
        """
        initial_cost = INSTANCE_MONTHLY_COST["m5.2xlarge"]
        cost_reduction_ratio = (initial_cost - state.monthly_cost) / initial_cost
        is_right_sized = state.instance_type in _SOLVED_INSTANCE_TYPES
        is_cheaper_enough = cost_reduction_ratio >= _COST_REDUCTION_THRESHOLD
        return is_right_sized and is_cheaper_enough
