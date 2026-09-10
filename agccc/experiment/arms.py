"""Arm registry (V3 §18).

Every arm receives the same task, the same agents, the same world, the same
suites and the same injection schedule. Only the coordination mechanism differs.
A fresh strategy instance is built per run because strategies carry per-run state
(the evidence model above all), and leaking that between runs would silently
make later seeds easier.
"""

from __future__ import annotations

from ..governance import policies
from ..governance.scoring import DEFAULT, ScoringConfig
from ..strategies.ablation import GovernedNonAdaptive, ObservationOnly
from ..strategies.adaptive import Adaptive
from ..strategies.static_graph import StaticGraph
from ..strategies.static_sequential import StaticSequential
from ..strategies.supervisor import Supervisor

ARM_ORDER = ("B1", "B2", "B3", "A1", "A2", "ABL2", "ABL4")

ARM_LABELS = {
    "B1": "B1: Static sequential",
    "B2": "B2: Supervisor",
    "B3": "B3: Static graph",
    "A1": "A1: AGCCC ungoverned",
    "A2": "Ours: AGCCC (full)",
    "ABL2": "Observation only",
    "ABL4": "Governed, non-adaptive",
}

# Table IV: (row label, SAL, AGL, re-orchestration, arm supplying the number)
ABLATION_ROWS = (
    ("Static graph", "No", "No", "No", "B3"),
    ("Observation only", "Yes", "No", "No", "ABL2"),
    ("Ungoverned adaptation", "Yes", "No", "Yes", "A1"),
    ("Governed, non-adaptive", "Yes", "Yes", "No", "ABL4"),
    (r"\textbf{Full AGCCC}", "Yes", "Yes", "Yes", "A2"),
)

# The five arms that appear in Table III.
MAIN_ARMS = ("B1", "B2", "B3", "A1", "A2")


def build(arm: str, scoring: ScoringConfig = DEFAULT, escalate_after: int = 2):
    reference = policies.default()
    if arm == "B1":
        return StaticSequential()
    if arm == "B2":
        return Supervisor(escalate_after=escalate_after)
    if arm == "B3":
        return StaticGraph()
    if arm == "A1":
        # Governance disabled: the filter stage runs with an empty policy set.
        # Scoring is identical to A2, so the arms differ by the filter alone.
        return Adaptive(policies=None, scoring=scoring, name="A1", reference_policies=reference)
    if arm == "A2":
        return Adaptive(
            policies=reference, scoring=scoring, name="A2", reference_policies=reference
        )
    if arm == "ABL2":
        return ObservationOnly(scoring=scoring)
    if arm == "ABL4":
        return GovernedNonAdaptive(
            policies=reference, scoring=scoring, reference_policies=reference
        )
    raise ValueError(f"unknown arm {arm!r}")
