"""Utility, cost, risk and the objective J.

    J(a) = P_success(a) · U(a)  −  λ_C · κ(a)  −  λ_R · R(a)

mapping onto the paper's Eq. 3 with the transition cost κ and the trade-off
weight λ made explicit per action (V3 §9).

This module never sees an inadmissible action. `decision.py` filters first and
passes only survivors, and a unit test asserts that a rejected candidate is
never scored — that ordering is Article #2's proposition, not an optimisation.

The defaults below are **[DECISION REQUIRED]** in V3 §9.2 and are reported as
chosen constants, not tuned values. λ_R in particular is swept in the
sensitivity analysis: the interesting question is whether a soft risk penalty
can substitute for a hard admissibility constraint, and the honest answer has to
be measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..awareness.confidence import W_FUNCTIONAL, W_SECURITY
from ..awareness.state import SituationState
from .actions import Action

# Expected additional agent invocations per action, used as the transition cost.
INVOCATION_COST: dict[Action, float] = {
    Action.ACCEPT: 0.0,
    Action.RETRY_CODER: 2.0,        # Coder + re-verify
    Action.INVOKE_SECURITY: 3.0,    # Security + re-verify, plus the graph edit
    Action.REPLAN: 4.0,             # Planner + Coder + re-verify
    Action.TERMINATE: 0.0,
}
COST_NORMALISER = 4.0


@dataclass(frozen=True)
class ScoringConfig:
    lambda_cost: float = 0.15
    lambda_risk: float = 0.50
    risk_accept_security_red: float = 1.0
    risk_accept_functional_red: float = 0.6
    risk_terminate: float = 0.3

    def hash(self) -> str:
        """V3 §19.1 — a run whose scoring hash differs from its experiment's
        registered hash is invalid and must be discarded, not reinterpreted."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


DEFAULT = ScoringConfig()


def utility(
    action: Action,
    state: SituationState,
    endpoints_implemented: int,
    units_total: int,
) -> float:
    """Expected final confidence *if the action succeeds*.

    Each repair action is credited with fixing the surface it targets and
    nothing else, so the utilities encode what an action is *for* rather than a
    hand-tuned preference between them.
    """
    f = state.functional.passed / state.functional.total if state.functional.total else 0.0
    s = state.security.passed / state.security.total if state.security.total else 0.0

    if action is Action.ACCEPT:
        return state.confidence
    if action is Action.RETRY_CODER:
        return W_FUNCTIONAL * 1.0 + W_SECURITY * s
    if action is Action.INVOKE_SECURITY:
        return W_FUNCTIONAL * f + W_SECURITY * 1.0
    if action is Action.REPLAN:
        # Re-planning only buys anything when the plan is missing coverage.
        missing = (units_total - endpoints_implemented) / units_total if units_total else 0.0
        return min(1.0, state.confidence + missing)
    return 0.0  # TERMINATE


def cost(action: Action) -> float:
    return INVOCATION_COST[action] / COST_NORMALISER


def risk(action: Action, state: SituationState, cfg: ScoringConfig = DEFAULT) -> float:
    if action is Action.ACCEPT:
        if state.security_failure:
            return cfg.risk_accept_security_red
        if state.functional_failure:
            return cfg.risk_accept_functional_red
        return 0.0
    if action is Action.TERMINATE:
        return cfg.risk_terminate
    return 0.0


def objective(
    action: Action,
    state: SituationState,
    p_success: float,
    endpoints_implemented: int,
    units_total: int,
    cfg: ScoringConfig = DEFAULT,
) -> float:
    return (
        p_success * utility(action, state, endpoints_implemented, units_total)
        - cfg.lambda_cost * cost(action)
        - cfg.lambda_risk * risk(action, state, cfg)
    )
