"""Runtime events.

Agents emit these; they never act on them. The separation is the architectural
claim under test (Raw_planning §12 Guideline 5): a Tester reports TEST_FAILED,
it does not decide ADD_SECURITY_AGENT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    TEST_FAILED = "TEST_FAILED"
    SECURITY_FAILURE = "SECURITY_FAILURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REQUIREMENT_CHANGED = "REQUIREMENT_CHANGED"
    CONTEXT_CHANGED = "CONTEXT_CHANGED"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TIMEOUT = "TIMEOUT"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    ALL_SUITES_GREEN = "ALL_SUITES_GREEN"
    GRAPH_MUTATED = "GRAPH_MUTATED"


@dataclass(frozen=True)
class Event:
    type: EventType
    source: str
    step: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "source": self.source,
            "step": self.step,
            "payload": self.payload,
        }


class EventBus:
    """Ordered, append-only event log with synchronous fan-out.

    Append-only because the event sequence is evidence: the reproducibility
    record (V3 §19.1) stores the full trace, and a mutable log would let a later
    step rewrite what an earlier step observed.
    """

    def __init__(self) -> None:
        self._log: list[Event] = []
        self._subscribers: list[Any] = []

    def subscribe(self, handler) -> None:
        self._subscribers.append(handler)

    def publish(self, event: Event) -> None:
        self._log.append(event)
        for handler in self._subscribers:
            handler(event)

    @property
    def log(self) -> tuple[Event, ...]:
        return tuple(self._log)

    def of_type(self, *types: EventType) -> tuple[Event, ...]:
        wanted = set(types)
        return tuple(e for e in self._log if e.type in wanted)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._log]
