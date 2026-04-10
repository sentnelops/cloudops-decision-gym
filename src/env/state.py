"""
CloudOps Decision Gym — State Encoding and Validation
Pure functions for converting EC2State to agent-consumable observations.
"""

from typing import Any

from constants import (
    INSTANCE_HIERARCHY,
    INSTANCE_MONTHLY_COST,
    MAX_MONTHLY_COST,
    VALID_IMDS_VERSIONS,
    VALID_ENVIRONMENTS,
)
from models import EC2State


class StateEncoder:
    """
    Converts EC2State into a flat observation dictionary suitable for agents.
    Numeric values are normalized to [0.0, 1.0] where applicable.
    Boolean and categorical fields are encoded as integers.
    """

    def to_observation(self, state: EC2State) -> dict[str, Any]:
        """
        Encode EC2State as a normalized observation dict.

        Keys:
            instance_idx        int   Position in INSTANCE_HIERARCHY (0 = smallest)
            instance_type       str   Raw instance type string (for logging/debug)
            cpu_avg_norm        float cpu_avg / 100.0
            cpu_p95_norm        float cpu_p95 / 100.0
            memory_avg_norm     float memory_avg / 100.0
            cost_norm           float monthly_cost / MAX_MONTHLY_COST
            internet_facing     int   1 if public, 0 if private
            ssh_open            int   1 if open, 0 if closed
            imds_v1             int   1 if IMDSv1 (risk present), 0 if IMDSv2
            is_prod             int   1 if production, 0 if dev
            is_overprovisioned  int   1 if CPU and memory both very low
            is_secure           int   1 if no known security issues
        """
        idx = INSTANCE_HIERARCHY.index(state.instance_type)
        return {
            "instance_idx": idx,
            "instance_type": state.instance_type,
            "cpu_avg_norm": round(state.cpu_avg / 100.0, 6),
            "cpu_p95_norm": round(state.cpu_p95 / 100.0, 6),
            "memory_avg_norm": round(state.memory_avg / 100.0, 6),
            "cost_norm": round(state.monthly_cost / MAX_MONTHLY_COST, 6),
            "internet_facing": int(state.internet_facing),
            "ssh_open": int(state.ssh_open),
            "imds_v1": int(state.imds_version == "v1"),
            "is_prod": int(state.environment == "prod"),
            "is_overprovisioned": int(state.is_overprovisioned),
            "is_secure": int(state.is_secure),
        }


class StateValidator:
    """
    Validates that an EC2State is internally consistent and within legal bounds.
    Raises ValueError with a clear message on the first violation found.
    """

    def validate(self, state: EC2State) -> None:
        """
        Check all fields for range and consistency violations.
        Raises ValueError if any check fails.
        """
        errors: list[str] = []

        if state.instance_type not in INSTANCE_HIERARCHY:
            errors.append(f"instance_type {state.instance_type!r} not in known hierarchy")

        if not (0.0 <= state.cpu_avg <= 100.0):
            errors.append(f"cpu_avg {state.cpu_avg} out of range [0, 100]")

        if not (0.0 <= state.cpu_p95 <= 100.0):
            errors.append(f"cpu_p95 {state.cpu_p95} out of range [0, 100]")

        if state.cpu_p95 < state.cpu_avg:
            errors.append(
                f"cpu_p95 ({state.cpu_p95}) cannot be less than cpu_avg ({state.cpu_avg})"
            )

        if not (0.0 <= state.memory_avg <= 100.0):
            errors.append(f"memory_avg {state.memory_avg} out of range [0, 100]")

        if state.monthly_cost < 0.0:
            errors.append(f"monthly_cost {state.monthly_cost} is negative")

        expected_cost = INSTANCE_MONTHLY_COST.get(state.instance_type)
        if expected_cost is not None and abs(state.monthly_cost - expected_cost) > 0.01:
            errors.append(
                f"monthly_cost {state.monthly_cost} does not match catalog value "
                f"{expected_cost} for {state.instance_type}"
            )

        if state.imds_version not in VALID_IMDS_VERSIONS:
            errors.append(f"imds_version {state.imds_version!r} must be one of {VALID_IMDS_VERSIONS}")

        if state.environment not in VALID_ENVIRONMENTS:
            errors.append(f"environment {state.environment!r} must be one of {VALID_ENVIRONMENTS}")

        if errors:
            raise ValueError("EC2State validation failed:\n  " + "\n  ".join(errors))
