"""Tier-1 simulated agents.

These are the *only* stubbed part of the system. The kernel, situation
awareness, governance and graph layers around them are the real implementation;
the plugin boundary in `agents.base` is what lets Tier 2 swap real LLM agents in
without touching anything else.

Each agent mutates the artefact according to the pre-drawn `World` and emits
events. None of them decides anything.
"""

from __future__ import annotations

from ...awareness.confidence import derive
from ...kernel.events import Event, EventType
from ...task.artefact import Artefact
from ...task.defects import SECURITY_DEFECTS
from ..base import (
    CODER,
    PLANNER,
    REVIEWER,
    SECURITY,
    TESTER,
    AgentContext,
    AgentResult,
)
from .world import World

LOW_CONFIDENCE_THRESHOLD = 0.75


class SimPlanner:
    name = PLANNER

    def __init__(self, world: World) -> None:
        self.world = world

    def act(self, ctx: AgentContext) -> AgentResult:
        planned = self.world.planned_endpoints(ctx.attempt)
        events = [
            Event(EventType.AGENT_COMPLETED, self.name, ctx.step, {"planned": len(planned)})
        ]
        if ctx.attempt > 0:
            events.append(Event(EventType.REQUIREMENT_CHANGED, self.name, ctx.step, {"replan": True}))
        return AgentResult(
            artefact=ctx.artefact,
            events=events,
            self_report=self.world.self_report(self.name, ctx.attempt),
            planned_endpoints=planned,
        )


class SimCoder:
    """Implements planned endpoints; strong on functional defects, weak on security."""

    name = CODER

    def __init__(self, world: World) -> None:
        self.world = world

    def act(self, ctx: AgentContext) -> AgentResult:
        artefact = ctx.artefact.with_endpoints(ctx.planned_endpoints)
        if ctx.attempt == 0:
            artefact = artefact.with_defects(self.world.initial_defects())
        else:
            cleared = self.world.coder_clears(ctx.attempt) & artefact.defects
            artefact = artefact.without_defects(cleared)
        return AgentResult(
            artefact=artefact,
            events=[
                Event(
                    EventType.AGENT_COMPLETED,
                    self.name,
                    ctx.step,
                    {"attempt": ctx.attempt, "defects": len(artefact.defects)},
                )
            ],
            self_report=self.world.self_report(self.name, ctx.attempt),
        )


class SimTester:
    """Runs the two real suites and reports. Decides nothing.

    This is the boundary the paper's argument rests on: the Tester turns an
    observation into *evidence*, and something else entirely turns evidence
    into a structural decision.
    """

    name = TESTER

    def __init__(self, world: World) -> None:
        self.world = world

    def act(self, ctx: AgentContext) -> AgentResult:
        reading = derive(
            ctx.artefact,
            llm_self_report=self.world.self_report(self.name, ctx.attempt),
        )
        f, s = reading.functional, reading.security
        events = [
            Event(
                EventType.AGENT_COMPLETED,
                self.name,
                ctx.step,
                {
                    "confidence": reading.value,
                    "functional": [f.passed, f.total],
                    "security": [s.passed, s.total],
                    "self_report": reading.llm_self_report,
                    "calibration_error": reading.calibration_error,
                },
            )
        ]
        if not f.green:
            events.append(
                Event(EventType.TEST_FAILED, self.name, ctx.step, {"failures": list(f.failures)})
            )
        if not s.green:
            events.append(
                Event(
                    EventType.SECURITY_FAILURE,
                    self.name,
                    ctx.step,
                    {"failures": list(s.failures)},
                )
            )
        if reading.value < LOW_CONFIDENCE_THRESHOLD:
            events.append(
                Event(EventType.LOW_CONFIDENCE, self.name, ctx.step, {"confidence": reading.value})
            )
        if f.green and s.green:
            events.append(Event(EventType.ALL_SUITES_GREEN, self.name, ctx.step, {}))
        return AgentResult(
            artefact=ctx.artefact,  # the Tester never changes the artefact
            events=events,
            self_report=reading.llm_self_report,
        )


class SimSecurity:
    """The specialist. Clears security defects the Coder cannot."""

    name = SECURITY

    def __init__(self, world: World) -> None:
        self.world = world

    def act(self, ctx: AgentContext) -> AgentResult:
        cleared = self.world.specialist_clears(ctx.attempt) & ctx.artefact.defects
        artefact = ctx.artefact.without_defects(cleared)
        return AgentResult(
            artefact=artefact,
            events=[
                Event(
                    EventType.AGENT_COMPLETED,
                    self.name,
                    ctx.step,
                    {"cleared": sorted(d.value for d in cleared)},
                )
            ],
            self_report=self.world.self_report(self.name, ctx.attempt),
        )


class SimReviewer:
    """Final gate. Reports whether the artefact is releasable; does not release it."""

    name = REVIEWER

    def __init__(self, world: World) -> None:
        self.world = world

    def act(self, ctx: AgentContext) -> AgentResult:
        reading = derive(ctx.artefact)
        releasable = reading.functional.green and reading.security.green
        return AgentResult(
            artefact=ctx.artefact,
            events=[
                Event(
                    EventType.AGENT_COMPLETED,
                    self.name,
                    ctx.step,
                    {"releasable": releasable, "confidence": reading.value},
                )
            ],
            self_report=self.world.self_report(self.name, ctx.attempt),
        )


def build_roster(world: World) -> dict[str, object]:
    return {
        a.name: a
        for a in (
            SimPlanner(world),
            SimCoder(world),
            SimTester(world),
            SimSecurity(world),
            SimReviewer(world),
        )
    }
