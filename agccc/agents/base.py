"""The agent contract.

One rule dominates this interface: **an agent may change the artefact and emit
events, and may do nothing else**. It cannot read policies, cannot see the
coordination graph, cannot choose who runs next, and cannot request its own
continuation. Raw_planning §12 Guideline 5 — a Tester reports TEST_FAILED; it
does not decide ADD_SECURITY_AGENT.

That constraint is enforced structurally: `AgentContext` simply does not carry
the graph or the governance engine, so there is nothing for an agent to reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..kernel.events import Event
from ..task.artefact import Artefact

PLANNER = "Planner"
CODER = "Coder"
TESTER = "Tester"
REVIEWER = "Reviewer"
SECURITY = "Security"


@dataclass
class AgentContext:
    """Everything an agent is permitted to see."""

    artefact: Artefact
    step: int
    attempt: int          # how many times this role has already run, 0-based
    planned_endpoints: frozenset = field(default_factory=frozenset)


@dataclass
class AgentResult:
    artefact: Artefact
    events: list[Event] = field(default_factory=list)
    self_report: float | None = None
    planned_endpoints: frozenset | None = None


class Agent(Protocol):
    name: str

    def act(self, ctx: AgentContext) -> AgentResult: ...
