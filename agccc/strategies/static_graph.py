"""B3 — static graph with conditional transitions.

The strongest non-adaptive baseline, and the one closest to how graph
orchestration frameworks are used in practice: the edge set is authored at
design time, includes the Security specialist in a fixed position, and carries a
bounded conditional retry edge back to the Coder.

    Planner -> Coder -> Tester -> Security -> Reviewer
                          ^_______|  (bounded retry, design-time constant)

Its limitation is structural rather than a matter of access. There is no
Security -> Tester edge, because no one authored one, so the specialist's work
is **never re-verified** and a second security pass is unreachable. Adding that
edge at runtime is exactly the operation B3 is defined not to have and AGCCC
performs with `ops.add_edge`.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, SECURITY, TESTER
from ..graph.model import CoordinationGraph
from .base import StrategyContext, StrategyDecision, Termination


class StaticGraph:
    name = "B3"
    adaptive = False

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def initial_graph(self) -> CoordinationGraph:
        nodes = frozenset({PLANNER, CODER, TESTER, SECURITY, REVIEWER})
        edges = frozenset(
            {
                (PLANNER, CODER),
                (CODER, TESTER),
                (TESTER, CODER),       # the design-time retry edge
                (TESTER, SECURITY),
                (SECURITY, REVIEWER),
                (TESTER, REVIEWER),
            }
        )
        return CoordinationGraph(nodes, edges, PLANNER)

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        state, g = ctx.state, ctx.graph

        if ctx.last_agent is None:
            return StrategyDecision(next_agent=PLANNER, graph=g)
        if ctx.last_agent == REVIEWER:
            return StrategyDecision(graph=g, terminate=Termination.SUCCESS)
        if ctx.last_agent == PLANNER:
            return StrategyDecision(next_agent=CODER, graph=g)
        if ctx.last_agent == CODER:
            return StrategyDecision(next_agent=TESTER, graph=g)
        if ctx.last_agent == SECURITY:
            # No Security -> Tester edge exists, so the fix is shipped unverified.
            return StrategyDecision(next_agent=REVIEWER, graph=g)

        # last_agent == TESTER
        if state.functional.green and state.security.green:
            return StrategyDecision(next_agent=REVIEWER, graph=g)

        coder_retries = max(0, state.attempts.get(CODER, 0) - 1)
        if coder_retries < self.max_retries and CODER in ctx.available:
            return StrategyDecision(next_agent=CODER, graph=g)
        if state.attempts.get(SECURITY, 0) == 0 and SECURITY in ctx.available:
            return StrategyDecision(next_agent=SECURITY, graph=g)
        return StrategyDecision(next_agent=REVIEWER, graph=g)
