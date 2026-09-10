"""Policy registry and evaluator.

Loads YAML policies and evaluates their `when` clauses against the situational
state. Deliberately dull: a policy can only *prohibit*. It cannot rank, score,
suggest, or prefer. Anything that expresses a preference belongs in scoring, and
keeping the two apart is what makes filter-then-rank checkable rather than
merely intended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..awareness.state import SituationState
from .actions import Action


@dataclass(frozen=True)
class Policy:
    id: str
    when: dict
    prohibit: frozenset[Action]
    description: str = ""

    def applies(self, state: SituationState) -> bool:
        for key, expected in self.when.items():
            if key == "confidence_below":
                if not state.confidence < expected:
                    return False
            elif key == "confidence_at_least":
                if not state.confidence >= expected:
                    return False
            else:
                if getattr(state, key) != expected:
                    return False
        return True


@dataclass(frozen=True)
class PolicySet:
    version: str
    policies: tuple[Policy, ...]

    def prohibited(self, state: SituationState) -> dict[Action, str]:
        """Action -> id of the first policy prohibiting it."""
        out: dict[Action, str] = {}
        for p in self.policies:
            if p.applies(state):
                for a in p.prohibit:
                    out.setdefault(a, p.id)
        return out


def load(path: str | Path) -> PolicySet:
    data = yaml.safe_load(Path(path).read_text())
    policies = tuple(
        Policy(
            id=p["id"],
            when=p.get("when", {}),
            prohibit=frozenset(Action(a) for a in p.get("prohibit", [])),
            description=p.get("description", ""),
        )
        for p in data["policies"]
    )
    return PolicySet(version=data["version"], policies=policies)


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "scenarios" / "authentication" / "policies.yaml"
)


def default() -> PolicySet:
    return load(DEFAULT_POLICY_PATH)


EMPTY = PolicySet(version="none", policies=())
