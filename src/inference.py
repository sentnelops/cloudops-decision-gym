"""
CloudOps Decision Gym — OpenEnv Inference Script (Phase 3)

Runs an LLM agent against a CloudOps Decision Gym task and logs results
in the strict OpenEnv format.

Environment variables:
    API_BASE_URL  Base URL for the OpenAI-compatible API
                  Default: https://api.openai.com/v1
    MODEL_NAME    Model identifier
                  Default: gpt-4o-mini
    HF_TOKEN      Hugging Face token — used as API key when set
                  (for HF Inference Endpoints / Spaces)
    TASK_NAME     Scenario to run: easy | medium | hard
                  Default: easy
    MAX_STEPS     Maximum steps per episode
                  Default: 8

Required log format (stdout):
    [START] task=<name> env=cloudops-decision-gym model=<model>
    [STEP]  step=<n> action=<ACTION> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Exit codes:
    0  Episode completed (success or timeout — not a script error)
    1  Fatal configuration or import error
"""

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "no-key-set")
TASK_NAME: str = os.environ.get("TASK_NAME", "easy")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "8"))

ENV_NAME: str = "cloudops-decision-gym"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a cloud infrastructure optimization agent managing an AWS EC2 instance.
Your goal is to make safe, cost-effective decisions.

## Priority order (highest to lowest):
1. SECURITY — Fix vulnerabilities before anything else
2. PERFORMANCE — Never downsize if CPU p95 > 60% or memory > 70%
3. COST — Reduce monthly cost when safe to do so

## Available actions (respond with EXACTLY one):
- DOWNSIZE      → Move to the next smaller instance type (reduces cost)
- UPSIZE        → Move to the next larger instance type (increases cost)
- FIX_SECURITY  → Upgrade IMDSv1→IMDSv2 and close SSH port 22
- NOOP          → Take no action

## Rules:
- If imds_version is "v1" OR ssh_open is true → you MUST call FIX_SECURITY first
- If cpu_p95 > 60 → do NOT DOWNSIZE (performance risk)
- If memory_avg > 70 → do NOT DOWNSIZE (memory risk)
- If instance is already small and cost is acceptable → NOOP is valid

## Response format:
Respond with ONLY valid JSON on a single line. No explanation, no markdown.
Example: {"action_type": "FIX_SECURITY"}
"""

_USER_TEMPLATE = """\
Current EC2 instance state:
{observation_json}

What is your next action? Respond with JSON only: {{"action_type": "<ACTION>"}}
"""

# ---------------------------------------------------------------------------
# Logging helpers (exact format required by OpenEnv spec)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _build_client():
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai package not installed. Run: pip install openai",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)


def call_llm(client: Any, observation_dict: dict) -> tuple[str, str | None]:
    """
    Call the LLM with the current observation.

    Returns:
        (action_type: str, error: str | None)
        action_type is always a valid action string (defaults to NOOP on parse failure).
    """
    user_message = _USER_TEMPLATE.format(
        observation_json=json.dumps(observation_dict, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,       # deterministic
            max_tokens=64,
            timeout=30,
        )
    except Exception as exc:
        return "NOOP", f"LLM API error: {type(exc).__name__}: {exc}"

    content = response.choices[0].message.content or ""
    return _parse_action(content)


def _parse_action(content: str) -> tuple[str, str | None]:
    """
    Parse the LLM response to extract an action_type.

    Strategy:
      1. Try JSON parse of full content
      2. Try to find a JSON object substring
      3. Scan for known action keywords
      4. Default to NOOP with an error message

    Returns:
        (action_type: str, error: str | None)
    """
    valid_actions = {"DOWNSIZE", "UPSIZE", "FIX_SECURITY", "NOOP"}
    content_stripped = content.strip()

    # Attempt 1: full JSON parse
    try:
        parsed = json.loads(content_stripped)
        if isinstance(parsed, dict) and "action_type" in parsed:
            candidate = str(parsed["action_type"]).upper().strip()
            if candidate in valid_actions:
                return candidate, None
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: find JSON object in content (LLM may wrap in markdown)
    start = content_stripped.find("{")
    end = content_stripped.rfind("}") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(content_stripped[start:end])
            if isinstance(parsed, dict) and "action_type" in parsed:
                candidate = str(parsed["action_type"]).upper().strip()
                if candidate in valid_actions:
                    return candidate, None
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: keyword scan (order matters — FIX_SECURITY before FIX)
    for action in ["FIX_SECURITY", "DOWNSIZE", "UPSIZE", "NOOP"]:
        if action in content_stripped.upper():
            return action, None

    # Fallback
    truncated = content_stripped[:80].replace("\n", " ")
    return "NOOP", f"unparseable response: {truncated!r}"


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run(task_name: str = TASK_NAME, max_steps: int = MAX_STEPS) -> bool:
    """
    Run one episode of the specified task using the configured LLM.

    Returns:
        True if the episode was solved, False if timed out.
    """
    # Imports here so the module is importable even without pydantic installed
    from env.environment import CloudOpsEnv
    from openenv_models import Action

    log_start(task=task_name, model=MODEL_NAME)

    client = _build_client()

    try:
        env = CloudOpsEnv(scenario=task_name)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    obs = env.reset()

    rewards_log: list[float] = []
    success = False
    steps_taken = 0

    for step_num in range(1, max_steps + 1):
        # Call LLM
        action_type, parse_error = call_llm(client, obs.to_prompt_dict())

        # Submit action to environment
        try:
            action = Action(action_type=action_type)
            obs, reward, done, info = env.step(action)
        except Exception as exc:
            # Unexpected environment error — log and abort
            log_step(step_num, action_type, 0.0, True, f"env error: {exc}")
            log_end(success=False, steps=step_num, rewards=rewards_log)
            return False

        rewards_log.append(reward.value)
        steps_taken = step_num
        log_step(step_num, action_type, reward.value, done, parse_error)

        if done:
            success = info.get("solved", False)
            break

    log_end(success=success, steps=steps_taken, rewards=rewards_log)
    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding task via CLI arg: python inference.py hard
    if len(sys.argv) > 1:
        task_arg = sys.argv[1]
    else:
        task_arg = TASK_NAME

    solved = run(task_name=task_arg)
    sys.exit(0)  # always exit 0 — script completed, result is in [END] line
