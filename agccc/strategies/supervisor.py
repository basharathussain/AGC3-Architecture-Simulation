"""B2 — centralised supervisor.

A single arbiter selects who acts next. Its roster **includes the Security
specialist**: without that, AGCCC would win by privileged access rather than by
architecture, and the comparison would be definitional rather than empirical.

What B2 still cannot do is the point of the comparison:

  * its topology is fixed — every agent is reachable from every other, and that
    never changes, so there is no G_{t+1} ≠ G_t;
  * its escalation is a **hand-authored constant**, `escalate_after`. It retries
    the Coder k times and then switches to the specialist, regardless of what
    the evidence says.

That constant is exactly the `if retry_count == k` that Raw_planning §12
Guideline 6 forbids AGCCC from having, and it is B2's defining weakness: a
well-chosen k performs well and a badly-chosen k wastes the budget, while the
right k differs per run. The sensitivity analysis sweeps k precisely so that
this claim is measured rather than asserted.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, SECURITY, TESTER
from ..graph.model import CoordinationGraph
from .base import StrategyContext, StrategyDecision, Termination

ROSTER = (PLANNER, CODER, TESTER, SECURITY, REVIEWER)


class Supervisor:
    name = "B2"
    adaptive = False

    def __init__(self, escalate_after: int = 2) -> None:
        self.escalate_after = escalate_after

    def initial_graph(self) -> CoordinationGraph:
        # Every agent reachable from every other: the arbiter may route
        # anywhere. Fixed for the whole run.
        edges = frozenset((a, b) for a in ROSTER for b in ROSTER if a != b)
        return CoordinationGraph(frozenset(ROSTER), edges, PLANNER)

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        state, g = ctx.state, ctx.graph

        if ctx.last_agent is None:
            return StrategyDecision(next_agent=PLANNER, graph=g)
        if ctx.last_agent == REVIEWER:
            return StrategyDecision(graph=g, terminate=Termination.SUCCESS)

        attempts = state.attempts
        if attempts.get(PLANNER, 0) == 0:
            return StrategyDecision(next_agent=PLANNER, graph=g)
        if attempts.get(CODER, 0) == 0:
            return StrategyDecision(next_agent=CODER, graph=g)

        # Anything that touched the artefact must be re-verified before the
        # arbiter can judge it.
        if ctx.last_agent in (CODER, SECURITY, PLANNER):
            if TESTER in ctx.available:
                return StrategyDecision(next_agent=TESTER, graph=g)
            return StrategyDecision(graph=g, terminate=Termination.NO_SUCCESSOR)

        # last_agent == TESTER: the arbiter's fixed selection rule.
        if state.functional.green and state.security.green:
            return StrategyDecision(next_agent=REVIEWER, graph=g)

        coder_retries = max(0, attempts.get(CODER, 0) - 1)
        target = CODER if coder_retries < self.escalate_after else SECURITY
        if target not in ctx.available:
            target = SECURITY if target == CODER else CODER
        if target not in ctx.available:
            return StrategyDecision(graph=g, terminate=Termination.NO_SUCCESSOR)
        return StrategyDecision(next_agent=target, graph=g)
