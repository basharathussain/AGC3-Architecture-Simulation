"""B4 — decentralised round-robin peers.

A coordination paradigm deliberately unlike the other four: no fixed pipeline, no
central arbiter, and no governed re-orchestration. Every agent is a peer on a
ring and the turn simply passes to the next available peer, with the Tester
interleaved so that work is verified as it accumulates.

This strategy exists to test the pluggability claim rather than to be a
competitive baseline. It was written after the microkernel was frozen, and the
measurement that matters is `git diff --stat agccc/kernel/` afterwards: if the
kernel had to learn anything about this paradigm, the claim that orchestration
lives outside the kernel would be false.
"""

from __future__ import annotations

from ..agents.base import CODER, PLANNER, REVIEWER, SECURITY, TESTER
from ..graph.model import CoordinationGraph
from .base import StrategyContext, StrategyDecision, Termination

RING = (PLANNER, CODER, TESTER, SECURITY, TESTER, REVIEWER)


class RoundRobin:
    name = "B4"
    adaptive = False

    def initial_graph(self) -> CoordinationGraph:
        nodes = frozenset(RING)
        edges = frozenset(zip(RING, RING[1:])) | {(RING[-1], RING[0])}
        return CoordinationGraph(nodes, edges, RING[0])

    def step(self, ctx: StrategyContext) -> StrategyDecision:
        if ctx.last_agent is None:
            return StrategyDecision(next_agent=RING[0], graph=ctx.graph)

        # Finish as soon as the ring has produced something releasable; the ring
        # itself carries no notion of admissibility, which is the point.
        if ctx.state.functional.green and ctx.state.security.green:
            if ctx.last_agent == REVIEWER:
                return StrategyDecision(graph=ctx.graph, terminate=Termination.SUCCESS)
            return StrategyDecision(next_agent=REVIEWER, graph=ctx.graph)

        if ctx.last_agent == REVIEWER:
            return StrategyDecision(graph=ctx.graph, terminate=Termination.SUCCESS)

        # Advance to the next peer on the ring that is currently available.
        i = RING.index(ctx.last_agent) if ctx.last_agent in RING else -1
        for offset in range(1, len(RING) + 1):
            candidate = RING[(i + offset) % len(RING)]
            if candidate in ctx.available:
                return StrategyDecision(next_agent=candidate, graph=ctx.graph)
        return StrategyDecision(graph=ctx.graph, terminate=Termination.NO_SUCCESSOR)
