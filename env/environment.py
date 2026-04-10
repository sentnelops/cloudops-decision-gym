"""
CloudOps Decision Gym — Core Environment
OpenEnv-compatible interface: reset / step / state.

Phase 2: CloudOpsEnv accepts a scenario name string ("easy", "medium", "hard")
         in addition to a scenario object instance. Backward-compatible.

Phase 3: Updated return types for full OpenEnv compliance.
         - reset()  → Observation (Pydantic)
         - step()   → (Observation, Reward, done, info)  [standard tuple]
         - step()   accepts either Pydantic Action OR internal Action IntEnum
         - state()  → EC2State  (unchanged — used by grader and internal tools)
"""

from typing import Any

from env.actions import Action as InternalAction
from env.actions import ActionSpace
from env.reward import RewardEngine
from env.state import StateEncoder, StateValidator
from models import EC2State
from openenv_models import (
    Action as OpenEnvAction,
    Observation,
    Reward,
)

# ---------------------------------------------------------------------------
# Action name → internal IntEnum mapping
# ---------------------------------------------------------------------------

_ACTION_NAME_TO_INTERNAL: dict[str, InternalAction] = {
    "DOWNSIZE": InternalAction.DOWNSIZE,
    "UPSIZE": InternalAction.UPSIZE,
    "FIX_SECURITY": InternalAction.FIX_SECURITY,
    "NOOP": InternalAction.NOOP,
}


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, Any]:
    from scenarios.ec2_cost_easy import EC2CostEasyScenario
    from scenarios.ec2_security_medium import EC2SecurityMediumScenario
    from scenarios.ec2_multistep_hard import EC2MultistepHardScenario
    return {
        "easy": EC2CostEasyScenario,
        "medium": EC2SecurityMediumScenario,
        "hard": EC2MultistepHardScenario,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CloudOpsEnv:
    """
    OpenEnv-compatible environment for cloud infrastructure decision-making.

    Phase 3 interface (recommended):
        env = CloudOpsEnv(scenario="easy")
        obs: Observation = env.reset()
        obs, reward, done, info = env.step(Action(action_type="DOWNSIZE"))

    Backward-compatible (Phase 1/2):
        env = CloudOpsEnv(scenario=EC2CostEasyScenario())
        obs = env.reset()                   # still works
        result = env.step(InternalAction.DOWNSIZE)  # still works

    step() accepts:
        - OpenEnvAction (Pydantic Action with action_type str)  — Phase 3
        - InternalAction (IntEnum DOWNSIZE/UPSIZE/etc.)         — Phase 1/2 compat

    step() returns:
        (Observation, Reward, done: bool, info: dict)

    state() returns EC2State — for grader and internal use.
    """

    def __init__(self, scenario: "str | Any") -> None:
        if isinstance(scenario, str):
            registry = _build_registry()
            if scenario not in registry:
                valid = list(registry.keys())
                raise ValueError(
                    f"Unknown scenario name {scenario!r}. Valid names: {valid}"
                )
            self.scenario = registry[scenario]()
        else:
            self.scenario = scenario

        self.max_steps: int = getattr(self.scenario, "max_steps", 10)

        self._action_space = ActionSpace()
        self._reward_engine = RewardEngine()
        self._encoder = StateEncoder()
        self._validator = StateValidator()

        self._current_state: EC2State | None = None
        self._step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to the scenario's initial state.
        Returns the initial Observation (Pydantic model).
        """
        initial = self.scenario.initial_state
        self._validator.validate(initial)
        self._current_state = initial
        self._step_count = 0
        self._done = False
        return self._to_observation(self._current_state)

    def step(
        self, action: "OpenEnvAction | InternalAction"
    ) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Apply an action to the current state.

        Args:
            action: Either an OpenEnvAction (Pydantic, from Phase 3 callers)
                    or an InternalAction IntEnum (from Phase 1/2 callers).

        Returns:
            (observation, reward, done, info)
            - observation: Observation (Pydantic)
            - reward:      Reward (Pydantic, value in [-1.0, 1.0])
            - done:        bool
            - info:        dict with step metadata and reward breakdown

        Raises:
            RuntimeError: If called before reset() or after episode is done.
            ValueError:   If action_type string is not a valid action name.
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        internal_action = self._resolve_action(action)

        prev_state = self._current_state
        self._step_count += 1

        # Apply action
        new_state, action_result = self._action_space.apply(internal_action, prev_state)

        # Validate resulting state
        self._validator.validate(new_state)

        # Compute reward
        reward_value = self._reward_engine.compute(prev_state, internal_action, new_state)
        reward_breakdown = self._reward_engine.breakdown(prev_state, internal_action, new_state)

        # Update internal state
        self._current_state = new_state

        # Determine done
        solved = self.scenario.is_solved(new_state)
        timed_out = self._step_count >= self.max_steps
        self._done = solved or timed_out

        observation = self._to_observation(new_state)
        reward = Reward(
            value=reward_value,
            breakdown={
                k: v for k, v in reward_breakdown.items()
                if k not in ("raw_total", "final_reward")
            },
        )

        info: dict[str, Any] = {
            "step": self._step_count,
            "action": internal_action.name,
            "action_applied": action_result.applied,
            "action_reason": action_result.reason,
            "reward_breakdown": reward_breakdown,
            "solved": solved,
            "timed_out": timed_out,
            "prev_instance_type": prev_state.instance_type,
            "next_instance_type": new_state.instance_type,
            "prev_cost": prev_state.monthly_cost,
            "next_cost": new_state.monthly_cost,
        }

        return observation, reward, self._done, info

    def state(self) -> EC2State:
        """
        Returns the current internal EC2State.
        Used by graders and Phase 1/2 code.
        Raises RuntimeError if called before reset().
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before state()")
        return self._current_state

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Human-readable summary of the current state."""
        if self._current_state is None:
            return "CloudOpsEnv: not initialized (call reset())"
        s = self._current_state
        scenario_name = type(self.scenario).__name__
        lines = [
            f"=== CloudOps Decision Gym [{scenario_name}] — Step {self._step_count}/{self.max_steps} ===",
            f"  Instance:     {s.instance_type}  (${s.monthly_cost:.2f}/mo)",
            f"  CPU:          avg={s.cpu_avg:.1f}%  p95={s.cpu_p95:.1f}%",
            f"  Memory:       avg={s.memory_avg:.1f}%",
            f"  Environment:  {s.environment}  (internet_facing={s.internet_facing})",
            f"  Security:     IMDS={s.imds_version}  SSH open={s.ssh_open}",
            f"  Secure:       {s.is_secure}  Overprovisioned: {s.is_overprovisioned}",
            f"  Done:         {self._done}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_action(
        self, action: "OpenEnvAction | InternalAction"
    ) -> InternalAction:
        """Convert OpenEnvAction or raw IntEnum to the internal InternalAction."""
        if isinstance(action, InternalAction):
            return action
        if isinstance(action, OpenEnvAction):
            name = action.action_type  # already uppercased by Pydantic validator
            return _ACTION_NAME_TO_INTERNAL[name]
        raise TypeError(
            f"action must be an OpenEnvAction or InternalAction, got {type(action).__name__}"
        )

    @staticmethod
    def _to_observation(state: EC2State) -> Observation:
        """
        Convert internal EC2State to the public Observation Pydantic model.
        Uses raw field values (not normalized) for human/LLM readability.
        """
        return Observation(
            instance_type=state.instance_type,
            cpu_avg=state.cpu_avg,
            cpu_p95=state.cpu_p95,
            memory_avg=state.memory_avg,
            monthly_cost=state.monthly_cost,
            internet_facing=state.internet_facing,
            ssh_open=state.ssh_open,
            imds_version=state.imds_version,
            environment=state.environment,
        )
