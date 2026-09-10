"""B1 — static sequential pipeline.

Planner -> Coder -> Tester -> Reviewer, once each, then stop. G_{t+1} = G_t for
every t. This is the weakest baseline and exists to establish the floor: it can
observe a failure and has no means whatsoever of responding to one.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, TESTER
from ..graph.model import CoordinationGraph, chain
from .base import StrategyContext, StrategyDecision, Termination


class StaticSequential:
    name = "B1"
    adaptive = False

    def initial_graph(self) -> CoordinationGraph:
        return chain(PLANNER, CODER, TESTER, REVIEWER)

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        if ctx.last_agent is None:
            return StrategyDecision(next_agent=ctx.graph.entry, graph=ctx.graph)

        successors = ctx.graph.successors(ctx.last_agent)
        if not successors:
            return StrategyDecision(graph=ctx.graph, terminate=Termination.NO_SUCCESSOR)

        nxt = successors[0]
        if nxt not in ctx.available:
            # The pipeline has no alternative route. An unavailable agent is
            # simply skipped, which is precisely the brittleness under test.
            after = ctx.graph.successors(nxt)
            if not after:
                return StrategyDecision(graph=ctx.graph, terminate=Termination.NO_SUCCESSOR)
            nxt = after[0]

        return StrategyDecision(next_agent=nxt, graph=ctx.graph)
