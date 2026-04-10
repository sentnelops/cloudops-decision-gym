"""
CloudOps Decision Gym — OpenEnv Typed Models (Phase 3)

Pydantic v2 models for the OpenEnv-compliant interface.

These models define the public contract between the environment and agents:
    Observation  — what the agent sees each step
    Action       — what the agent can submit
    Reward       — what the environment returns after each action

Naming note: `Action` here is a Pydantic model (action_type: str).
The internal discrete action space lives in env/actions.py as an IntEnum.
"""

from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    """
    Raw (un-normalized) observation of an EC2 instance's current state.
    All fields use real-world units for human and LLM readability.
    """

    instance_type: str    # e.g. "m5.2xlarge"
    cpu_avg: float        # average CPU utilization, percent [0.0, 100.0]
    cpu_p95: float        # 95th-percentile CPU utilization, percent [0.0, 100.0]
    memory_avg: float     # average memory utilization, percent [0.0, 100.0]
    monthly_cost: float   # estimated monthly cost in USD (e.g. 276.48)
    internet_facing: bool # True if publicly reachable
    ssh_open: bool        # True if port 22 open to 0.0.0.0/0
    imds_version: str     # "v1" (vulnerable) or "v2" (secure)
    environment: str      # "prod" or "dev"

    model_config = {"frozen": True}

    def to_prompt_dict(self) -> dict:
        """Returns a clean dict suitable for inclusion in an LLM prompt."""
        return {
            "instance_type": self.instance_type,
            "cpu_avg_pct": self.cpu_avg,
            "cpu_p95_pct": self.cpu_p95,
            "memory_avg_pct": self.memory_avg,
            "monthly_cost_usd": self.monthly_cost,
            "internet_facing": self.internet_facing,
            "ssh_open": self.ssh_open,
            "imds_version": self.imds_version,
            "environment": self.environment,
        }


class Action(BaseModel):
    """
    Action submitted by the agent to the environment.

    action_type must be one of: DOWNSIZE, UPSIZE, FIX_SECURITY, NOOP
    Case-insensitive on input — always stored as uppercase.
    """

    action_type: str

    @field_validator("action_type", mode="before")
    @classmethod
    def normalise_and_validate(cls, v: str) -> str:
        valid = {"DOWNSIZE", "UPSIZE", "FIX_SECURITY", "NOOP"}
        normalised = str(v).upper().strip()
        if normalised not in valid:
            raise ValueError(
                f"action_type {v!r} is not valid. Must be one of: {sorted(valid)}"
            )
        return normalised

    model_config = {"frozen": True}


class Reward(BaseModel):
    """
    Reward returned by the environment after each step.

    value:     scalar reward in [-1.0, 1.0]
    breakdown: per-component reward values for diagnostics
    """

    value: float
    breakdown: dict[str, float] = {}

    model_config = {"frozen": True}
