---
title: CloudOps Decision Gym
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.12"
app_file: main.py
pinned: false
---

# CloudOps Decision Gym

An open-source environment for training AI agents to make safe, cost-effective cloud infrastructure decisions.

Built on the [OpenEnv](https://openenv.dev) specification. Compatible with any LLM or RL agent via a standard `reset / step / state` interface.

---

## What Is This?

CloudOps Decision Gym simulates the decisions an on-call engineer makes when managing AWS EC2 instances:

- Is this instance overprovisioned? Should I downsize it?
- Are there active security vulnerabilities? Should I fix them first?
- What are the risks of resizing a production machine right now?

Agents must learn that **security always comes before cost** — and that production environments raise the stakes on every wrong decision.

---

## Key Features

| Feature | Description |
|---|---|
| **OpenEnv compliant** | Standard `reset()` / `step()` / `state()` interface |
| **3 difficulty tiers** | easy, medium, hard — progressively complex decision chains |
| **Multi-objective reward** | Cost savings, security improvement, performance risk, sequencing penalties |
| **Constraint engine** | Explicit, named rule violations with production environment amplification |
| **Deterministic** | No randomness — reproducible results across runs |
| **LLM inference script** | Drop-in `inference.py` with OpenAI-compatible API support |
| **Interactive UI** | Gradio app deployable to Hugging Face Spaces |
| **Typed models** | Pydantic v2 `Observation`, `Action`, `Reward` for clean agent interfaces |

---

## Project Structure

```
cloudops-decision-gym/
│
├── cloudops_decision_gym/          # Core library
│   ├── constants.py                # Instance catalog, costs, reward coefficients
│   ├── models.py                   # EC2State, StepResult (internal dataclasses)
│   ├── openenv_models.py           # Observation, Action, Reward (Pydantic / public API)
│   │
│   ├── env/
│   │   ├── environment.py          # CloudOpsEnv — main entry point
│   │   ├── actions.py              # Action enum + ActionSpace (pure state transitions)
│   │   ├── state.py                # StateEncoder, StateValidator
│   │   ├── reward.py               # RewardEngine (5-component reward)
│   │   └── constraints.py         # ConstraintEngine (security-first rules)
│   │
│   ├── scenarios/
│   │   ├── ec2_cost_easy.py        # Easy: overprovisioned dev instance
│   │   ├── ec2_security_medium.py  # Medium: security conflict
│   │   └── ec2_multistep_hard.py   # Hard: prod with amplified penalties
│   │
│   └── graders/
│       └── ec2_grader.py           # grade_episode() → deterministic [0.0, 1.0] score
│
├── openenv.yaml                    # OpenEnv specification
├── inference.py                    # LLM inference loop
├── main.py                         # Root entry point for deployment
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/cloudops-decision-gym.git
cd cloudops-decision-gym

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install pydantic pyyaml       # core only
pip install openai                # for inference.py
pip install gradio                # for app.py
```

> Requires Python 3.12+

### 2. Run your first episode

```python
from cloudops_decision_gym.env.environment import CloudOpsEnv
from cloudops_decision_gym.openenv_models import Action
from cloudops_decision_gym.graders.ec2_grader import grade_episode_detailed
from cloudops_decision_gym.env.actions import Action as InternalAction

# Load a scenario
env = CloudOpsEnv(scenario="easy")
obs = env.reset()
initial_state = env.state()

print(f"Starting: {obs.instance_type} at ${obs.monthly_cost:.2f}/mo, CPU {obs.cpu_avg}%")

# Run an episode
actions_taken = []
while True:
    obs, reward, done, info = env.step(Action(action_type="DOWNSIZE"))
    actions_taken.append(InternalAction.DOWNSIZE)
    print(f"  Step {info['step']}: {info['action']} → reward {reward.value:+.4f}")
    if done:
        break

# Grade the episode
score = grade_episode_detailed(actions_taken, env.state(), initial_state)
print(f"\nFinal score: {score['total_score']:.4f}")
print(f"  Cost reduction:      {score['cost_reduction']:.4f}")
print(f"  Safety maintained:   {score['safety_maintained']:.4f}")
print(f"  Sequence correct:    {score['sequence_correctness']:.4f}")
```

**Expected output:**
```
Starting: m5.2xlarge at $276.48/mo, CPU 7.0%
  Step 1: DOWNSIZE → reward +0.1832
  Step 2: DOWNSIZE → reward +0.1584
  Step 3: DOWNSIZE → reward -0.0613
  Step 4: DOWNSIZE → reward +0.1098

Final score: 0.9451
  Cost reduction:      0.7803
  Safety maintained:   1.0000
  Sequence correct:    1.0000
```

### 3. Launch the interactive UI

```bash
python main.py
# → Open http://localhost:7860
```

### 4. Run with an LLM

```bash
export OPENAI_API_KEY="sk-..."
export TASK_NAME="medium"        # easy | medium | hard
export MODEL_NAME="gpt-4o-mini"

python inference.py
```

**Output format:**
```
[START] task=medium env=cloudops-decision-gym model=gpt-4o-mini
[STEP] step=1 action=FIX_SECURITY reward=0.50 done=false error=null
[STEP] step=2 action=DOWNSIZE reward=0.16 done=true error=null
[END] success=true steps=2 rewards=0.50,0.16
```

---

## The Environment

### Observation Space

Each step returns an `Observation` object with these fields:

| Field | Type | Description |
|---|---|---|
| `instance_type` | `str` | EC2 instance type, e.g. `"m5.2xlarge"` |
| `cpu_avg` | `float` | Average CPU utilization, % |
| `cpu_p95` | `float` | 95th-percentile CPU utilization, % |
| `memory_avg` | `float` | Average memory utilization, % |
| `monthly_cost` | `float` | Estimated monthly cost, USD |
| `internet_facing` | `bool` | True if publicly reachable |
| `ssh_open` | `bool` | True if port 22 open to `0.0.0.0/0` |
| `imds_version` | `str` | `"v1"` (vulnerable) or `"v2"` (secure) |
| `environment` | `str` | `"prod"` or `"dev"` |

### Action Space

4 discrete actions:

| Action | Effect |
|---|---|
| `DOWNSIZE` | Move to the next smaller instance type in the hierarchy |
| `UPSIZE` | Move to the next larger instance type |
| `FIX_SECURITY` | Upgrade IMDSv1 → IMDSv2 and close SSH port 22 |
| `NOOP` | Take no action |

### Instance Hierarchy

Instances are ordered smallest → largest. DOWNSIZE/UPSIZE move one step in this sequence:

```
t3.micro ($8.47) → t3.small ($16.93) → t3.medium ($30.37) → t3.large ($60.74)
→ t3.xlarge ($121.47) → m5.large ($87.60) → m5.xlarge ($175.20)
→ m5.2xlarge ($276.48) → m5.4xlarge ($552.96)
```

> Note: The hierarchy is **capacity-ordered**, not cost-ordered. Moving from `m5.large` to `t3.xlarge` increases cost — agents must learn this.

---

## Scenarios

### Easy — `ec2_cost_easy`

> Pure cost optimization. No security issues.

| Field | Value |
|---|---|
| Instance | `m5.2xlarge` — $276.48/mo |
| CPU avg / p95 | 7% / 12% |
| Security | Clean (IMDSv2, SSH closed) |
| Optimal sequence | `DOWNSIZE` (repeat until t3.large or smaller) |
| Solved when | Instance ≤ t3.large AND cost reduced ≥ 40% |
| Max steps | 5 |

### Medium — `ec2_security_medium`

> Security conflict. Agent must learn to fix vulnerabilities before optimizing cost.

| Field | Value |
|---|---|
| Instance | `m5.xlarge` — $175.20/mo |
| CPU avg / p95 | 12% / 18% |
| Security | IMDSv1 active, SSH open, internet-facing |
| Optimal sequence | `FIX_SECURITY` → `DOWNSIZE` |
| Solved when | `is_secure` AND cost reduced ≥ 20% |
| Max steps | 7 |

**Wrong sequence penalty:** Downsizing before fixing security incurs `−0.40` sequencing penalty per step.

### Hard — `ec2_multistep_hard`

> Production environment. All penalties amplified 1.5×.

| Field | Value |
|---|---|
| Instance | `m5.2xlarge` — $276.48/mo |
| CPU avg / p95 | 35% / 52% |
| Security | IMDSv1 active, SSH open, internet-facing |
| Environment | **prod** |
| Optimal sequence | `FIX_SECURITY` → `DOWNSIZE` |
| Solved when | `is_secure` AND cost reduced ≥ 20% |
| Max steps | 8 |

**Production multiplier:** All constraint penalties are multiplied by `1.5×`. Wrong sequence penalty becomes `−0.60`.

---

## Reward Function

The reward is a scalar in `[−1.0, +1.0]`, computed each step as the sum of 5 components:

```
reward = cost_saving_score
       + security_improvement_score
       + performance_risk_penalty      (≤ 0)
       + unsafe_action_penalty         (≤ 0)
       + sequencing_penalty            (≤ 0)
```

### Components

| Component | Signal | Detail |
|---|---|---|
| `cost_saving_score` | `+` | `(prev_cost − next_cost) / max_catalog_cost`, range `[−0.2, +1.0]` |
| `security_improvement_score` | `+` | `+0.30` for IMDSv1→v2, `+0.20` for closing SSH |
| `performance_risk_penalty` | `−` | `−0.50` if DOWNSIZE when `cpu_p95 > 60%`; `−0.30` if `memory_avg > 70%` |
| `unsafe_action_penalty` | `−` | `−0.80` for UPSIZE on overprovisioned instance; `−0.10` for NOOP when obviously needed |
| `sequencing_penalty` | `−` | `−0.40` for DOWNSIZE before FIX_SECURITY; `−0.25` for NOOP while insecure; `×1.5` in prod |

### Reward breakdown

Every `step()` call includes a `reward.breakdown` dict:

```python
obs, reward, done, info = env.step(Action(action_type="FIX_SECURITY"))
print(reward.breakdown)
# {
#   'cost_saving_score': 0.0,
#   'security_improvement_score': 0.5,
#   'performance_risk_penalty': 0.0,
#   'unsafe_action_penalty': 0.0,
#   'sequencing_penalty': 0.0
# }
```

---

## Grader

The grader evaluates a complete episode and returns a score in `[0.0, 1.0]`:

```python
from cloudops_decision_gym.graders.ec2_grader import grade_episode_detailed
from cloudops_decision_gym.env.actions import Action as InternalAction

score = grade_episode_detailed(
    actions=[InternalAction.FIX_SECURITY, InternalAction.DOWNSIZE],
    final_state=env.state(),
    initial_state=initial_state,
)
# {
#   'cost_reduction':       0.5000,  # weight 0.25
#   'safety_maintained':    1.0000,  # weight 0.25
#   'sequence_correctness': 1.0000,  # weight 0.25
#   'no_risky_actions':     1.0000,  # weight 0.15
#   'efficiency':           1.0000,  # weight 0.10
#   'total_score':          0.8750
# }
```

### Grading criteria

| Criterion | Weight | How it scores |
|---|---|---|
| `cost_reduction` | 0.25 | `(initial_cost − final_cost) / initial_cost`, clamped `[0, 1]` |
| `safety_maintained` | 0.25 | `1.0` if no security regressions; `0.0` if any regression occurred |
| `sequence_correctness` | 0.25 | `1.0` correct order; `0.5` fixed eventually but too late; `0.0` never fixed |
| `no_risky_actions` | 0.15 | `1.0` if no DOWNSIZE under high load; `0.0` if risky downsize taken |
| `efficiency` | 0.10 | `1.0 − 0.2×(noop_count)`, floor `0.0` |

### Score interpretation

| Score | Interpretation |
|---|---|
| `0.90 – 1.00` | Optimal or near-optimal decisions |
| `0.70 – 0.89` | Safe but suboptimal (e.g. wrong sequence but eventually corrected) |
| `0.40 – 0.69` | Significant issues (missed security fix, risky downsizes) |
| `0.00 – 0.39` | Dangerous decisions (ignored security, wrong-sequence in prod) |

---

## Integration Guide

### Option 1: Use the Pydantic interface (recommended)

```python
from cloudops_decision_gym.env.environment import CloudOpsEnv
from cloudops_decision_gym.openenv_models import Action, Observation, Reward

env = CloudOpsEnv(scenario="medium")
obs: Observation = env.reset()

# Your agent decides
action = Action(action_type="FIX_SECURITY")  # case-insensitive

obs, reward, done, info = env.step(action)
# obs    → Observation (Pydantic model with .to_prompt_dict())
# reward → Reward (Pydantic model with .value and .breakdown)
# done   → bool
# info   → dict with step metadata
```

### Option 2: Use the IntEnum interface (for RL frameworks)

```python
from cloudops_decision_gym.env.environment import CloudOpsEnv
from cloudops_decision_gym.env.actions import Action

env = CloudOpsEnv(scenario="hard")
obs = env.reset()

obs, reward, done, info = env.step(Action.FIX_SECURITY)
obs, reward, done, info = env.step(Action.DOWNSIZE)
```

### Option 3: String scenario name vs scenario object

Both are valid:

```python
# String (Phase 3+ recommended)
env = CloudOpsEnv(scenario="hard")

# Object (Phase 1 style, still supported)
from cloudops_decision_gym.scenarios.ec2_multistep_hard import EC2MultistepHardScenario
env = CloudOpsEnv(scenario=EC2MultistepHardScenario())
```

### Building a custom agent

```python
from cloudops_decision_gym.env.environment import CloudOpsEnv
from cloudops_decision_gym.openenv_models import Action, Observation
from cloudops_decision_gym.env.actions import Action as InternalAction
from cloudops_decision_gym.graders.ec2_grader import grade_episode_detailed


def my_agent(obs: Observation) -> str:
    """Return an action_type string based on observation."""
    # Example: always fix security first
    if obs.imds_version == "v1" or obs.ssh_open:
        return "FIX_SECURITY"
    if obs.cpu_avg < 20:
        return "DOWNSIZE"
    return "NOOP"


def run(scenario: str) -> dict:
    env = CloudOpsEnv(scenario=scenario)
    obs = env.reset()
    initial_state = env.state()
    actions_taken = []

    while True:
        action_str = my_agent(obs)
        obs, reward, done, info = env.step(Action(action_type=action_str))
        actions_taken.append(InternalAction[action_str])
        if done:
            break

    return grade_episode_detailed(actions_taken, env.state(), initial_state)


for scenario in ["easy", "medium", "hard"]:
    result = run(scenario)
    print(f"{scenario}: {result['total_score']:.4f}")
```

### LLM agent via `inference.py`

Set environment variables and run:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o"
export TASK_NAME="hard"
python inference.py

# Hugging Face Inference Endpoints
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TASK_NAME="medium"
python inference.py

# Any OpenAI-compatible endpoint (Ollama, vLLM, etc.)
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="llama3.2"
python inference.py
```

### Adding a custom scenario

Create a new file in `cloudops_decision_gym/scenarios/`:

```python
# scenarios/ec2_my_scenario.py
from cloudops_decision_gym.constants import INSTANCE_MONTHLY_COST
from cloudops_decision_gym.models import EC2State


class MyScenario:
    description = "Your scenario description."
    max_steps = 10

    @property
    def initial_state(self) -> EC2State:
        return EC2State(
            instance_type="m5.xlarge",
            cpu_avg=60.0,
            cpu_p95=78.0,       # above 60% → DOWNSIZE is risky
            memory_avg=55.0,
            monthly_cost=INSTANCE_MONTHLY_COST["m5.xlarge"],
            internet_facing=False,
            ssh_open=False,
            imds_version="v2",
            environment="prod",
        )

    def is_solved(self, state: EC2State) -> bool:
        # Define your own completion condition
        return state.monthly_cost < 100.0
```

Then use it directly:

```python
from cloudops_decision_gym.env.environment import CloudOpsEnv
from scenarios.ec2_my_scenario import MyScenario

env = CloudOpsEnv(scenario=MyScenario())
```

Or register it in the string registry by editing `_build_registry()` in [env/environment.py](cloudops_decision_gym/env/environment.py).

---

## Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Hardware: CPU Basic (free tier works)

2. Add a `requirements.txt` at the project root:

   ```
   gradio>=4.0
   pydantic>=2.0
   pyyaml>=6.0
   ```

3. Push the repository. The Space will automatically run `main.py`.

The `main.py` launches with `server_name="0.0.0.0"` which is required for Spaces.

---

## Architecture Reference

```
                        ┌─────────────────────────────────┐
                        │         CloudOpsEnv             │
                        │  reset() → Observation          │
                        │  step(Action) →                 │
                        │    (Observation, Reward,        │
                        │     done, info)                 │
                        │  state() → EC2State             │
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┼────────────────────────┐
              │                      │                        │
     ┌────────▼─────────┐  ┌─────────▼────────┐  ┌──────────▼──────────┐
     │   ActionSpace    │  │  RewardEngine     │  │  StateValidator     │
     │  apply(action,   │  │  compute() →      │  │  validate(state)    │
     │   state) →       │  │   float           │  │                     │
     │  (new_state,     │  │  breakdown() →    │  └─────────────────────┘
     │   result)        │  │   dict            │
     └──────────────────┘  └─────────┬─────────┘
                                     │
                           ┌─────────▼─────────┐
                           │  ConstraintEngine  │
                           │  evaluate(state,   │
                           │   action) →        │
                           │  [Violation, ...]  │
                           └────────────────────┘

     Scenarios                Grader                 Pydantic Models
     ──────────               ──────────             ───────────────
     initial_state            grade_episode()        Observation
     is_solved()              grade_episode_         Action
     max_steps                  detailed()           Reward
```

---

## Design Principles

**1. Deterministic by design**
No `random` calls anywhere. Scenario initial states are fixed literals. Running the same scenario with the same action sequence always produces the same rewards and scores.

**2. State immutability**
`EC2State` is a frozen Pydantic dataclass. Every action returns a *new* state object — no mutation in place. This makes it safe to snapshot, replay, or branch episodes.

**3. Security-first signal**
The reward function and constraint engine are designed so that ignoring security vulnerabilities is *never* the right move. On a prod instance with IMDSv1 active, downsizing before fixing security gives `−0.60` per step — strong enough to dominate the cost saving signal.

**4. Modular components**
Each subsystem (actions, reward, constraints, grader) is a standalone class with no shared mutable state. You can use `RewardEngine` or `ConstraintEngine` directly in your own evaluation pipeline without needing the full environment.

**5. OpenEnv compatibility**
The `reset() / step() / state()` interface maps directly to the OpenEnv specification. The `openenv.yaml` file fully describes the environment for automated tooling.

---

## Contributing

This project is intentionally kept simple to remain easy to understand and extend.

**Adding a new scenario:** See [Adding a custom scenario](#adding-a-custom-scenario) above.

**Adding a new action:** Add to the `Action` IntEnum in [env/actions.py](cloudops_decision_gym/env/actions.py), implement `_apply_<action>` in `ActionSpace`, and add reward/constraint logic as needed.

**Adding a new cloud resource type (e.g. RDS, S3):** Create a new state model in `models.py`, a new action space in `env/`, and new scenarios. The grader and reward engine are resource-agnostic and can be reused.

Pull requests welcome. Please keep new code deterministic and dependency-free (no cloud SDKs, no ML frameworks in the core library).

---

## License

Apache 2.0 — see `LICENSE`.

---

## Citation

If you use CloudOps Decision Gym in your research, please cite:

```bibtex
@software{cloudops_decision_gym,
  title  = {CloudOps Decision Gym},
  year   = {2026},
  url    = {https://github.com/your-org/cloudops-decision-gym},
  note   = {An OpenEnv-compliant environment for training AI agents on cloud infrastructure decisions}
}
```
