"""Table IV ablation arms — isolating what each layer is worth.

    #  Configuration           SAL  AGL  Re-orch.
    1  Static graph            No   No   No        -> B3
    2  Observation only        Yes  No   No        -> ObservationOnly (here)
    3  Ungoverned adaptation   Yes  No   Yes       -> A1
    4  Governed, non-adaptive  Yes  Yes  No        -> GovernedNonAdaptive (here)
    5  Full AGCCC              Yes  Yes  Yes       -> A2

The two arms defined here are real executions, not rows copied from a
neighbouring configuration.

`ObservationOnly` is the genuinely interesting one. It runs the whole situation
awareness stack *and* scores every candidate, then throws the answer away and
follows the fixed pipeline anyway. It exists to make the point that observation
without control is worth exactly nothing — and the honest way to establish that
is to measure it rather than to assume it.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, TESTER
from ..governance.actions import Action
from ..governance.decision import DecisionEngine
from ..governance.evidence import EvidenceModel
from ..governance.policies import PolicySet
from ..governance.scoring import DEFAULT, ScoringConfig
from ..graph.model import CoordinationGraph, chain
from .adaptive import Adaptive
from .base import StrategyContext, StrategyDecision, Termination


class ObservationOnly:
    """SAL yes, AGL no, re-orchestration no.

    Situation awareness is fully active — the runtime folds every event into
    state exactly as it does for AGCCC — and the decision engine is consulted at
    every Tester step. Its answer is recorded for the trace and then discarded.
    Execution follows the static chain regardless.
    """

    name = "ABL2"
    adaptive = False

    def __init__(self, scoring: ScoringConfig = DEFAULT) -> None:
        self.engine = DecisionEngine(policies=None, scoring=scoring)
        self.evidence = EvidenceModel()
        self.observed_but_ignored: list[dict] = []

    def initial_graph(self) -> CoordinationGraph:
        return chain(PLANNER, CODER, TESTER, REVIEWER)

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        if ctx.last_agent is None:
            return StrategyDecision(next_agent=ctx.graph.entry, graph=ctx.graph)

        if ctx.last_agent == TESTER:
            d = self.engine.decide(
                ctx.state, self.evidence, len(ctx.artefact.endpoints), ctx.units_total
            )
            self.observed_but_ignored.append(
                {
                    "step": ctx.step,
                    "would_have_selected": d.selected.value if d.selected else None,
                    "confidence": round(ctx.state.confidence, 6),
                }
            )

        successors = ctx.graph.successors(ctx.last_agent)
        if not successors:
            return StrategyDecision(graph=ctx.graph, terminate=Termination.NO_SUCCESSOR)
        return StrategyDecision(next_agent=successors[0], graph=ctx.graph)


class GovernedNonAdaptive(Adaptive):
    """SAL yes, AGL yes, re-orchestration no.

    The policy filter is fully active, so an unsafe release is structurally
    unreachable, but the action space contains no structural repair. The arm can
    refuse; it cannot fix. Completion should therefore be poor while compliance
    is perfect — which is the safety/capability trade-off the full architecture
    is meant to resolve rather than merely balance.
    """

    name = "ABL4"

    def __init__(
        self,
        policies: PolicySet,
        scoring: ScoringConfig = DEFAULT,
        reference_policies: PolicySet | None = None,
    ) -> None:
        super().__init__(
            policies=policies,
            scoring=scoring,
            name="ABL4",
            reference_policies=reference_policies,
        )
        self.engine = DecisionEngine(
            policies=policies,
            scoring=scoring,
            action_space=(Action.ACCEPT, Action.TERMINATE),
        )
