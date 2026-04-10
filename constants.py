"""
CloudOps Decision Gym — Constants
Instance catalog, cost data, and reward coefficients.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Instance type hierarchy — ordered smallest → largest (used for DOWNSIZE/UPSIZE)
# ---------------------------------------------------------------------------

INSTANCE_HIERARCHY: list[str] = [
    "t3.micro",    # index 0
    "t3.small",    # index 1
    "t3.medium",   # index 2
    "t3.large",    # index 3
    "t3.xlarge",   # index 4
    "m5.large",    # index 5
    "m5.xlarge",   # index 6
    "m5.2xlarge",  # index 7
    "m5.4xlarge",  # index 8
]

# Monthly on-demand cost (USD) per instance type (us-east-1 approximations)
INSTANCE_MONTHLY_COST: Dict[str, float] = {
    "t3.micro":   8.47,
    "t3.small":   16.93,
    "t3.medium":  30.37,
    "t3.large":   60.74,
    "t3.xlarge":  121.47,
    "m5.large":   87.60,
    "m5.xlarge":  175.20,
    "m5.2xlarge": 276.48,
    "m5.4xlarge": 552.96,
}

# Maximum possible monthly cost (top of hierarchy — used for reward normalization)
MAX_MONTHLY_COST: float = INSTANCE_MONTHLY_COST["m5.4xlarge"]

# ---------------------------------------------------------------------------
# Performance thresholds
# ---------------------------------------------------------------------------

CPU_SAFE_MAX: float = 70.0       # CPU avg above this is "at risk"
CPU_CRITICAL_MAX: float = 85.0   # CPU avg above this is "critical load"
CPU_P95_DOWNSIZE_LIMIT: float = 60.0  # p95 above this makes DOWNSIZE risky
MEMORY_SAFE_MAX: float = 80.0    # Memory avg above this is "at risk"
MEMORY_DOWNSIZE_LIMIT: float = 70.0   # Memory above this makes DOWNSIZE risky

# "Overprovisioned" thresholds — low utilization signals waste
CPU_OVERPROVISION_MAX: float = 20.0   # CPU avg below this = overprovisioned
MEMORY_OVERPROVISION_MAX: float = 30.0

# ---------------------------------------------------------------------------
# Reward coefficients
# ---------------------------------------------------------------------------

REWARD_COST_SAVING_WEIGHT: float = 0.6
REWARD_SECURITY_WEIGHT: float = 0.4

# Reward component bounds (before final clamp)
COST_SAVING_SCORE_MAX: float = 1.0
COST_SAVING_SCORE_MIN: float = -0.2   # penalize cost increases

SECURITY_IMDS_UPGRADE: float = 0.30   # reward for upgrading IMDSv1 → v2
SECURITY_SSH_CLOSE: float = 0.20      # reward for closing open SSH

PERFORMANCE_RISK_P95_PENALTY: float = -0.50   # downsize with high p95
PERFORMANCE_RISK_MEM_PENALTY: float = -0.30   # downsize with high memory

UNSAFE_UPSIZE_PENALTY: float = -0.80    # upsize already-overprovisioned instance
NOOP_OBVIOUS_PENALTY: float = -0.10     # noop when action is clearly needed

# Phase 2 — sequencing and security constraint penalties
SEQUENCING_PENALTY_BASE: float = -0.40   # DOWNSIZE before FIX_SECURITY when issues exist
SECURITY_IGNORE_PENALTY: float = -0.25   # NOOP when security issues are present
PROD_RISK_MULTIPLIER: float = 1.5        # scale all constraint penalties in prod environment
PROD_SEQUENCING_PENALTY: float = -0.60   # heavier sequencing penalty in prod (= BASE × MULTIPLIER)

# Final reward range
REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Episode config defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS: int = 10

# ---------------------------------------------------------------------------
# Valid field values
# ---------------------------------------------------------------------------

VALID_IMDS_VERSIONS: tuple[str, ...] = ("v1", "v2")
VALID_ENVIRONMENTS: tuple[str, ...] = ("prod", "dev")
VALID_CPU_RANGE: tuple[float, float] = (0.0, 100.0)
VALID_MEMORY_RANGE: tuple[float, float] = (0.0, 100.0)
