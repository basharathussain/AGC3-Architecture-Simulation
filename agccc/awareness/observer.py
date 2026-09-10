"""Observer — folds the event stream into `SituationState`.

This module is allowed to *classify* (a SECURITY_FAILURE event means the
security surface is red, which means deployment is not allowed) and forbidden to
*decide* (nothing here selects an action). It imports nothing from `governance`,
`graph` or `strategies`, and that direction of dependency is checked by a test.
"""

from __future__ import annotations

from dataclasses import replace

from ..kernel.events import Event, EventType
from ..task.artefact import Artefact
from .confidence import derive
from .state import SituationState, SuiteView, TaskStatus


class Observer:
    def __init__(self) -> None:
        self._state = SituationState()

    @property
    def state(self) -> SituationState:
        return self._state

    def note_invocation(self, agent: str, step: int) -> None:
        attempts = dict(self._state.attempts)
        attempts[agent] = attempts.get(agent, 0) + 1
        self._state = replace(
            self._state,
            attempts=attempts,
            current_agent=agent,
            step=step,
            security_agent_active=self._state.security_agent_active or agent == "Security",
        )

    def note_unavailable(self, agent: str, unavailable: bool) -> None:
        current = set(self._state.unavailable)
        current.add(agent) if unavailable else current.discard(agent)
        self._state = replace(self._state, unavailable=frozenset(current))

    def observe(self, event: Event) -> None:
        s = self._state
        if event.type is EventType.AGENT_COMPLETED and event.source == "Planner":
            s = replace(s, task_status=TaskStatus.PLANNED)
        elif event.type is EventType.AGENT_COMPLETED and event.source == "Coder":
            s = replace(s, task_status=TaskStatus.IMPLEMENTED)
        elif event.type is EventType.AGENT_COMPLETED and event.source == "Tester":
            p = event.payload
            s = replace(
                s,
                confidence=p["confidence"],
                functional=SuiteView(*p["functional"]),
                security=SuiteView(*p["security"]),
                tested_at_least_once=True,
            )
        elif event.type is EventType.TEST_FAILED:
            s = replace(s, functional_failure=True, task_status=TaskStatus.VERIFICATION_FAILED)
        elif event.type is EventType.SECURITY_FAILURE:
            s = replace(
                s,
                security_failure=True,
                deployment_allowed=False,
                task_status=TaskStatus.VERIFICATION_FAILED,
            )
        elif event.type is EventType.LOW_CONFIDENCE:
            s = replace(s, low_confidence=True)
        elif event.type is EventType.ALL_SUITES_GREEN:
            s = replace(
                s,
                task_status=TaskStatus.VERIFIED,
                security_failure=False,
                functional_failure=False,
                low_confidence=False,
                deployment_allowed=True,
            )
        elif event.type is EventType.AGENT_UNAVAILABLE:
            s = replace(s, unavailable=s.unavailable | {event.source})
        self._state = s

    def resync(self, artefact: Artefact) -> None:
        """Recompute suite-derived fields directly from the artefact.

        The event fold is the primary path; this is a consistency check used by
        the runtime at termination so that a reported final confidence is never
        a stale event payload. Both paths must agree — a test asserts it.
        """
        reading = derive(artefact)
        self._state = replace(
            self._state,
            confidence=reading.value,
            functional=SuiteView(reading.functional.passed, reading.functional.total),
            security=SuiteView(reading.security.passed, reading.security.total),
        )
