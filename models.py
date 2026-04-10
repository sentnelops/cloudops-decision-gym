"""
CloudOps Decision Gym — Data Models
Core dataclasses used across the environment.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EC2State:
    """
    Immutable snapshot of an EC2 instance's operational state.
    Frozen to enforce state immutability — all transitions return new instances.
    """
    instance_type: str          # e.g. "m5.2xlarge"
    cpu_avg: float              # average CPU utilization (%)
    cpu_p95: float              # 95th percentile CPU utilization (%)
    memory_avg: float           # average memory utilization (%)
    monthly_cost: float         # estimated monthly cost in USD
    internet_facing: bool       # is the instance publicly reachable?
    ssh_open: bool              # is port 22 open to 0.0.0.0/0?
    imds_version: str           # "v1" or "v2"
    environment: str            # "prod" or "dev"

    def __post_init__(self) -> None:
        from constants import (
            VALID_IMDS_VERSIONS, VALID_ENVIRONMENTS, INSTANCE_HIERARCHY
        )
        if self.instance_type not in INSTANCE_HIERARCHY:
            raise ValueError(f"Unknown instance_type: {self.instance_type!r}")
        if self.imds_version not in VALID_IMDS_VERSIONS:
            raise ValueError(f"imds_version must be one of {VALID_IMDS_VERSIONS}")
        if self.environment not in VALID_ENVIRONMENTS:
            raise ValueError(f"environment must be one of {VALID_ENVIRONMENTS}")
        if not (0.0 <= self.cpu_avg <= 100.0):
            raise ValueError(f"cpu_avg must be in [0, 100], got {self.cpu_avg}")
        if not (0.0 <= self.cpu_p95 <= 100.0):
            raise ValueError(f"cpu_p95 must be in [0, 100], got {self.cpu_p95}")
        if not (0.0 <= self.memory_avg <= 100.0):
            raise ValueError(f"memory_avg must be in [0, 100], got {self.memory_avg}")
        if self.monthly_cost < 0.0:
            raise ValueError(f"monthly_cost must be non-negative, got {self.monthly_cost}")

    @property
    def is_secure(self) -> bool:
        """True if no known security risks."""
        return self.imds_version == "v2" and not self.ssh_open

    @property
    def is_overprovisioned(self) -> bool:
        """True if utilization is very low relative to instance capacity."""
        from constants import CPU_OVERPROVISION_MAX, MEMORY_OVERPROVISION_MAX
        return self.cpu_avg < CPU_OVERPROVISION_MAX and self.memory_avg < MEMORY_OVERPROVISION_MAX


@dataclass
class StepResult:
    """
    Result of a single environment step.
    Mirrors the (obs, reward, done, info) tuple from standard RL environments.
    """
    observation: dict[str, Any]   # agent-consumable observation dict
    reward: float                  # scalar reward for this step
    done: bool                     # whether the episode has ended
    info: dict[str, Any] = field(default_factory=dict)  # diagnostic metadata
