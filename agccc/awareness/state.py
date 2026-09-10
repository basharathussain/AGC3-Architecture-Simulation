"""The situational state — the answer to "what is happening?".

Deliberately contains no answer to "what should happen next?". It holds
`security_failure: True` and `deployment_allowed: False`; it does not hold
"invoke the security agent". That separation is the reason this is a state
object and not a rule, and it is what Raw_planning §4 contrasts against
`if test_failed: retry()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    PLANNED = "planned"
    IMPLEMENTED = "implemented"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIED = "verified"


@dataclass(frozen=True)
class SuiteView:
    passed: int = 0
    total: int = 0

    @property
    def green(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def to_dict(self) -> dict:
        return {"passed": self.passed, "total": self.total}


@dataclass(frozen=True)
class SituationState:
    step: int = 0
    current_agent: str | None = None
    task_status: TaskStatus = TaskStatus.NOT_STARTED
    confidence: float = 0.0
    functional: SuiteView = field(default_factory=SuiteView)
    security: SuiteView = field(default_factory=SuiteView)
    security_failure: bool = False
    functional_failure: bool = False
    low_confidence: bool = False
    deployment_allowed: bool = False
    tested_at_least_once: bool = False
    attempts: dict[str, int] = field(default_factory=dict)
    unavailable: frozenset[str] = frozenset()
    security_agent_active: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "current_agent": self.current_agent,
            "task_status": self.task_status.value,
            "confidence": round(self.confidence, 6),
            "functional_tests": self.functional.to_dict(),
            "security_tests": self.security.to_dict(),
            "security_failure": self.security_failure,
            "functional_failure": self.functional_failure,
            "low_confidence": self.low_confidence,
            "deployment_allowed": self.deployment_allowed,
            "attempts": dict(sorted(self.attempts.items())),
            "unavailable": sorted(self.unavailable),
            "security_agent_active": self.security_agent_active,
        }
