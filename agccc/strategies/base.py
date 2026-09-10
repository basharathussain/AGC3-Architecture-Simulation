"""The coordination-strategy plugin boundary.

Every arm is a plugin behind this interface. H5 (V3 §17) is the claim that
adding a strategy requires **zero** change to `agccc/kernel/`, and it is
verified by `git diff --stat`, not by assertion — so this interface has to be
sufficient for a strategy the kernel has never heard of.

A strategy answers exactly one question per step: *who runs next, and what is
the coordination graph now?* It returns a new graph rather than mutating one,
so the kernel can diff G_t against G_{t+1} and log the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..awareness.state import SituationState
from ..graph.model import CoordinationGraph
from ..kernel.audit import AdaptationRecord
from ..task.artefact import Artefact


class Termination(str, Enum):
    SUCCESS = "success"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ESCALATED = "escalated"
    NO_SUCCESSOR = "no_successor"


@dataclass
class StrategyContext:
    graph: CoordinationGraph
    state: SituationState
    artefact: Artefact
    step: int
    last_agent: str | None
    available: frozenset[str]


@dataclass
class StrategyDecision:
    next_agent: str | None = None
    graph: CoordinationGraph | None = None
    audit: AdaptationRecord | None = None
    terminate: Termination | None = None


class Strategy(Protocol):
    name: str
    adaptive: bool

    def initial_graph(self) -> CoordinationGraph: ...

    def step(self, ctx: StrategyContext) -> StrategyDecision: ...
