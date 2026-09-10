"""A2 (governed) and A1 (ungoverned) — the AGCCC adaptive strategy.

The two arms are *the same class*, differing only in whether a policy set is
supplied to the decision engine. Nothing else changes: same candidates, same
utilities, same costs, same risks, same evidence model, same graph operations.
Any measured difference between A1 and A2 is therefore attributable to the
admissibility filter and to nothing else, which is what makes H3 a test rather
than a demonstration.

Execution is genuinely graph-driven. Every node except the Tester has exactly
one successor, so the runtime follows edges; the Tester is the only ambiguous
node, and that ambiguity is resolved by governance. Adaptation is expressed as
real structural change:

    RETRY_CODER      +edge Tester -> Coder
    INVOKE_SECURITY  +node Security, +edge Tester -> Security, +edge Security -> Tester
    REPLAN           +edge Tester -> Planner
    ACCEPT           follow the existing Tester -> Reviewer edge

None of these is "pick a different agent": each changes E_t, and the first
changes V_t as well.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, SECURITY, TESTER
from ..governance.actions import Action
from ..governance.decision import DecisionEngine
from ..governance.evidence import EvidenceModel
from ..governance.policies import PolicySet
from ..governance.scoring import DEFAULT, ScoringConfig
from ..graph import ops
from ..graph.diff import diff
from ..graph.model import CoordinationGraph, chain
from ..kernel.audit import AdaptationRecord
from .base import StrategyContext, StrategyDecision, Termination


class Adaptive:
    adaptive = True

    def __init__(
        self,
        policies: PolicySet | None = None,
        scoring: ScoringConfig = DEFAULT,
        name: str = "A2",
        reference_policies: PolicySet | None = None,
    ) -> None:
        self.name = name
        self.engine = DecisionEngine(policies=policies, scoring=scoring)
        self.evidence = EvidenceModel()
        # The yardstick for metric 9, "actions executed that were later found
        # policy-violating". A1 enforces nothing but is still *measured* against
        # the same policy set A2 enforces — otherwise H3 would have nothing to
        # compare. This set is never consulted by the filter; it only records.
        self.reference_policies = reference_policies
        self.violations: list[dict] = []
        self._pending: Action | None = None
        self._confidence_before: float = 0.0
        self.decisions: list[dict] = []

    def initial_graph(self) -> CoordinationGraph:
        return chain(PLANNER, CODER, TESTER, REVIEWER)

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        if ctx.last_agent is None:
            return StrategyDecision(next_agent=ctx.graph.entry, graph=ctx.graph)

        if ctx.last_agent == REVIEWER:
            return StrategyDecision(graph=ctx.graph, terminate=Termination.SUCCESS)

        if ctx.last_agent != TESTER:
            successors = [s for s in ctx.graph.successors(ctx.last_agent) if s in ctx.available]
            if not successors:
                return StrategyDecision(graph=ctx.graph, terminate=Termination.NO_SUCCESSOR)
            return StrategyDecision(next_agent=successors[0], graph=ctx.graph)

        return self._govern(ctx)

    # --- the decision point ---------------------------------------------

    def _govern(self, ctx: StrategyContext) -> StrategyDecision:
        state = ctx.state

        # Close the loop on the previous action before choosing the next one.
        # "Success" is measured, not assumed: did the artefact actually improve?
        if self._pending is not None:
            improved = state.confidence > self._confidence_before
            self.evidence.record(self._pending, improved)

        decision = self.engine.decide(
            state=state,
            evidence=self.evidence,
            endpoints_implemented=len(ctx.artefact.endpoints),
        )

        trigger = (
            "SECURITY_FAILURE"
            if state.security_failure
            else "TEST_FAILED"
            if state.functional_failure
            else "ALL_SUITES_GREEN"
        )
        record = AdaptationRecord(
            step=ctx.step,
            trigger=trigger,
            confidence=state.confidence,
            candidates=[a.value for a in decision.considered],
            rejected={a.value: pid for a, pid in decision.rejected.items()},
            scores={a.value: v for a, v in decision.scores.items()},
            selected=decision.selected.value if decision.selected else None,
        )
        self.decisions.append(
            {
                "step": ctx.step,
                "confidence": round(state.confidence, 6),
                "p_success": {a.value: round(v, 6) for a, v in decision.p_success.items()},
                "scores": {a.value: round(v, 6) for a, v in decision.scores.items()},
                "rejected": {a.value: pid for a, pid in decision.rejected.items()},
                "selected": decision.selected.value if decision.selected else None,
            }
        )

        if decision.escalated:
            # Escalate, never relax. No policy is weakened to keep going.
            record.graph_before = record.graph_after = str(ctx.graph)
            record.state_after = state.to_dict()
            return StrategyDecision(
                graph=ctx.graph, audit=record, terminate=Termination.ESCALATED
            )

        action = decision.selected

        if self.reference_policies is not None:
            breached = self.reference_policies.prohibited(state).get(action)
            if breached is not None:
                self.violations.append(
                    {"step": ctx.step, "action": action.value, "policy": breached}
                )

        graph, nxt, terminate = self._apply(action, ctx)

        record.graph_before = str(ctx.graph)
        record.graph_after = str(graph)
        record.graph_diff = diff(ctx.graph, graph).to_dict()
        record.state_after = state.to_dict()

        self._pending = action if action not in (Action.ACCEPT, Action.TERMINATE) else None
        self._confidence_before = state.confidence

        return StrategyDecision(next_agent=nxt, graph=graph, audit=record, terminate=terminate)

    def _apply(self, action: Action, ctx: StrategyContext):
        """Action -> graph transformation. The only place topology changes."""
        g = ctx.graph
        if action is Action.ACCEPT:
            return g, REVIEWER, None
        if action is Action.TERMINATE:
            return g, None, Termination.ESCALATED
        if action is Action.RETRY_CODER:
            return ops.add_edge(g, TESTER, CODER), CODER, None
        if action is Action.REPLAN:
            return ops.add_edge(g, TESTER, PLANNER), PLANNER, None
        if action is Action.INVOKE_SECURITY:
            g = ops.add_edge(ops.add_node(g, SECURITY), TESTER, SECURITY)
            g = ops.add_edge(g, SECURITY, TESTER)
            return g, SECURITY, None
        raise ValueError(f"unhandled action {action!r}")
