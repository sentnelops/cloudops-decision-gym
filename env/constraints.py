"""
CloudOps Decision Gym — Constraint Engine
Encodes environment-aware decision rules as explicit, named violations.

The ConstraintEngine is called by the RewardEngine before computing reward.
Each rule that fires produces a ConstraintViolation with a penalty value.
Penalties are summed and added as the sequencing_penalty reward component.

Rules (evaluated in order):
    security_first      DOWNSIZE while security issues exist
    security_ignore     NOOP while security issues exist
    safe_downsize       DOWNSIZE when cpu_p95 is above the safe limit

Prod amplification:
    All base penalties are multiplied by PROD_RISK_MULTIPLIER when
    the instance environment is "prod".
"""

from dataclasses import dataclass

from constants import (
    CPU_P95_DOWNSIZE_LIMIT,
    SEQUENCING_PENALTY_BASE,
    SECURITY_IGNORE_PENALTY,
    PROD_RISK_MULTIPLIER,
)
from env.actions import Action
from models import EC2State


@dataclass(frozen=True)
class ConstraintViolation:
    """A single rule violation produced by the ConstraintEngine."""
    rule: str        # short identifier, e.g. "security_first"
    penalty: float   # negative float applied to reward
    message: str     # human-readable explanation


class ConstraintEngine:
    """
    Evaluates a (state, action) pair against all constraint rules.
    Returns a list of ConstraintViolation objects (empty list = no violations).

    Usage:
        engine = ConstraintEngine()
        violations = engine.evaluate(prev_state, action)
        total_penalty = sum(v.penalty for v in violations)
    """

    def evaluate(self, state: EC2State, action: Action) -> list[ConstraintViolation]:
        """
        Check all rules for the given (state, action) pair.
        Returns violations found. Empty list means all constraints satisfied.
        """
        violations: list[ConstraintViolation] = []
        multiplier = PROD_RISK_MULTIPLIER if state.environment == "prod" else 1.0

        v = self._check_security_first(state, action, multiplier)
        if v:
            violations.append(v)

        v = self._check_security_ignore(state, action, multiplier)
        if v:
            violations.append(v)

        v = self._check_safe_downsize(state, action, multiplier)
        if v:
            violations.append(v)

        return violations

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_security_first(
        self, state: EC2State, action: Action, multiplier: float
    ) -> ConstraintViolation | None:
        """
        Rule: DOWNSIZE is not allowed while security issues exist.
        Security issues = IMDSv1 active OR SSH port 22 open to public.

        Rationale: a vulnerable instance should be hardened before any
        sizing change that might require a restart or network reconfiguration.
        """
        if action != Action.DOWNSIZE:
            return None
        if state.is_secure:
            return None

        base_penalty = SEQUENCING_PENALTY_BASE
        penalty = base_penalty * multiplier

        issues: list[str] = []
        if state.imds_version == "v1":
            issues.append("IMDSv1 active")
        if state.ssh_open:
            issues.append("SSH open")

        env_note = " (prod — penalty amplified)" if state.environment == "prod" else ""
        return ConstraintViolation(
            rule="security_first",
            penalty=round(penalty, 4),
            message=(
                f"DOWNSIZE attempted while security issues exist: "
                f"{', '.join(issues)}{env_note}"
            ),
        )

    def _check_security_ignore(
        self, state: EC2State, action: Action, multiplier: float
    ) -> ConstraintViolation | None:
        """
        Rule: NOOP is penalized when security issues are present.
        Doing nothing while the instance is vulnerable is negligent.
        """
        if action != Action.NOOP:
            return None
        if state.is_secure:
            return None

        penalty = SECURITY_IGNORE_PENALTY * multiplier
        env_note = " (prod — penalty amplified)" if state.environment == "prod" else ""
        return ConstraintViolation(
            rule="security_ignore",
            penalty=round(penalty, 4),
            message=(
                f"NOOP while security issues exist "
                f"(IMDS={state.imds_version}, SSH={state.ssh_open}){env_note}"
            ),
        )

    def _check_safe_downsize(
        self, state: EC2State, action: Action, multiplier: float
    ) -> ConstraintViolation | None:
        """
        Rule: DOWNSIZE under high CPU p95 load.
        This rule overlaps with the performance_risk_penalty in RewardEngine,
        but the constraint version applies the prod multiplier, making prod
        environments more sensitive to risky downsizes.

        Only fires in prod — dev already handled by RewardEngine directly.
        """
        if action != Action.DOWNSIZE:
            return None
        if state.environment != "prod":
            return None  # dev handled by base performance_risk_penalty
        if state.cpu_p95 <= CPU_P95_DOWNSIZE_LIMIT:
            return None

        # In prod, amplify the existing performance risk
        additional_penalty = -0.20 * multiplier  # stacks on top of perf penalty
        return ConstraintViolation(
            rule="safe_downsize",
            penalty=round(additional_penalty, 4),
            message=(
                f"DOWNSIZE in prod with cpu_p95={state.cpu_p95:.1f}% "
                f"(limit={CPU_P95_DOWNSIZE_LIMIT}%) — amplified penalty applied"
            ),
        )
