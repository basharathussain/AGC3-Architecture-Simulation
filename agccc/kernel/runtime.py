"""The microkernel runtime.

Small and boring, by design (Raw_planning §2). It contains no planning logic, no
security logic, no retry rules, no prompts, and no hard-coded agent sequence. It
provides infrastructure only: execute the selected agent, publish its events,
let situation awareness fold them into state, ask the active strategy what the
graph and the next agent are, diff the graph, log any mutation.

Because orchestration is not embedded here, the same runtime hosts all five
arms unmodified — which is the H5 claim, verified by diffing this directory
after a sixth strategy is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext
from ..awareness.observer import Observer
from ..graph.diff import diff
from ..graph.model import CoordinationGraph
from ..strategies.base import Strategy, StrategyContext, Termination
from ..task.artefact import EMPTY, Artefact
from .audit import AuditLogger
from .events import Event, EventBus, EventType

DEFAULT_STEP_BUDGET = 24


@dataclass
class RunRecord:
    arm: str
    env: str
    seed: int
    outcome: Termination
    steps: int
    invocations: dict[str, int]
    graph_history: list[str]
    adaptations: list[dict]
    events: list[dict]
    final_artefact: dict
    final_confidence: float
    functional: tuple[int, int]
    security: tuple[int, int]
    audit_completeness: float
    decision_latency_ms: float
    execution_ms: float
    governance_violations: int
    calibration_errors: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.functional[0] == self.functional[1] and self.security[0] == self.security[1]

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "env": self.env,
            "seed": self.seed,
            "outcome": self.outcome.value,
            "success": self.success,
            "steps": self.steps,
            "invocations": dict(sorted(self.invocations.items())),
            "graph_history": self.graph_history,
            "adaptations": self.adaptations,
            "final_artefact": self.final_artefact,
            "final_confidence": round(self.final_confidence, 6),
            "functional": list(self.functional),
            "security": list(self.security),
            "audit_completeness": self.audit_completeness,
            "decision_latency_ms": round(self.decision_latency_ms, 4),
            "execution_ms": round(self.execution_ms, 4),
            "governance_violations": self.governance_violations,
            "calibration_errors": [round(c, 6) for c in self.calibration_errors],
            "events": self.events,
            "extra": self.extra,
        }


class Runtime:
    def __init__(
        self,
        roster: dict[str, Any],
        strategy: Strategy,
        injector=None,
        step_budget: int = DEFAULT_STEP_BUDGET,
        units_total: int = 0,
        spec=None,
    ) -> None:
        self.roster = roster
        self.strategy = strategy
        self.injector = injector
        self.step_budget = step_budget
        # A plain count, not a task object: the kernel still names no unit,
        # role or policy (asserted by test_the_kernel_contains_no_orchestration_logic).
        self.units_total = units_total
        self.bus = EventBus()
        self.observer = Observer(spec)
        self.audit = AuditLogger()

    def run(self, arm: str, env: str, seed: int) -> RunRecord:
        self.bus.subscribe(self.observer.observe)

        artefact: Artefact = EMPTY
        graph: CoordinationGraph = self.strategy.initial_graph()
        graph_history = [str(graph)]
        planned: frozenset = frozenset()
        last_agent: str | None = None
        outcome = Termination.BUDGET_EXHAUSTED
        decision_ms = 0.0
        started = time.perf_counter()
        step = 0

        while step < self.step_budget:
            available = self._available(graph, step)

            t0 = time.perf_counter()
            decision = self.strategy.step(
                StrategyContext(
                    graph=graph,
                    state=self.observer.state,
                    artefact=artefact,
                    step=step,
                    last_agent=last_agent,
                    available=available,
                    units_total=self.units_total,
                )
            )
            decision_ms += (time.perf_counter() - t0) * 1000

            # Every governance decision is audited, whether or not it moved the
            # topology. A decision that deliberately left the graph alone is
            # still a decision, and the E1 trace needs the scores from all of
            # them (V3 §20).
            if decision.audit is not None:
                self.audit.record(decision.audit)

            new_graph = decision.graph or graph
            if new_graph != graph:
                d = diff(graph, new_graph)
                self.bus.publish(
                    Event(EventType.GRAPH_MUTATED, "kernel", step, {"diff": str(d)})
                )
                graph = new_graph
                graph_history.append(str(graph))

            if decision.terminate is not None:
                outcome = decision.terminate
                break
            if decision.next_agent is None:
                outcome = Termination.NO_SUCCESSOR
                break

            agent_name = decision.next_agent
            if agent_name not in self.roster:
                raise KeyError(f"strategy selected an unregistered agent: {agent_name!r}")

            self.observer.note_invocation(agent_name, step)
            ctx = AgentContext(
                artefact=artefact,
                step=step,
                attempt=self.observer.state.attempts.get(agent_name, 1) - 1,
                planned_endpoints=planned,
            )
            result = self.roster[agent_name].act(ctx)

            artefact = result.artefact
            if self.injector is not None:
                perturbed = self.injector.perturb(artefact, step)
                if perturbed != artefact:
                    self.bus.publish(
                        Event(EventType.CONTEXT_CHANGED, "injector", step, {"regression": True})
                    )
                    artefact = perturbed
            if result.planned_endpoints is not None:
                planned = result.planned_endpoints
            for e in result.events:
                self.bus.publish(e)

            last_agent = agent_name
            step += 1

            if self.observer.state.task_status.value == "verified":
                outcome = Termination.SUCCESS
                break

        execution_ms = (time.perf_counter() - started) * 1000
        self.observer.resync(artefact)
        state = self.observer.state

        return RunRecord(
            arm=arm,
            env=env,
            seed=seed,
            outcome=outcome,
            steps=step,
            invocations=dict(state.attempts),
            graph_history=graph_history,
            adaptations=self.audit.to_list(),
            events=self.bus.to_list(),
            final_artefact=artefact.to_dict(),
            final_confidence=state.confidence,
            functional=(state.functional.passed, state.functional.total),
            security=(state.security.passed, state.security.total),
            audit_completeness=self.audit.completeness,
            decision_latency_ms=decision_ms,
            execution_ms=execution_ms,
            governance_violations=self._violations(),
            calibration_errors=self._calibration_errors(),
        )

    # --- helpers --------------------------------------------------------

    def _available(self, graph: CoordinationGraph, step: int) -> frozenset[str]:
        everyone = frozenset(self.roster)
        if self.injector is None:
            return everyone
        down = self.injector.unavailable_at(step)
        for name in down:
            self.observer.note_unavailable(name, True)
            self.bus.publish(Event(EventType.AGENT_UNAVAILABLE, name, step, {}))
        return everyone - frozenset(down)

    def _violations(self) -> int:
        """Actions executed that policy would have forbidden (V3 §21 metric 9).

        Counted post-hoc from the trace by the ungoverned/governed comparison in
        `experiment.metrics`; the kernel records zero and defers, because the
        kernel does not know what the policy set is.
        """
        return 0

    def _calibration_errors(self) -> list[float]:
        """Collect Mode C vs Mode B divergence from whichever agent reports it.

        Keyed on the payload rather than on an agent name: the kernel must not
        know which role happens to run the suites, or it would stop being a
        microkernel the moment someone renamed one.
        """
        return [
            e.payload["calibration_error"]
            for e in self.bus.log
            if e.payload.get("calibration_error") is not None
        ]
