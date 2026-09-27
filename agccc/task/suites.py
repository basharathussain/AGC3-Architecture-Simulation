"""The two acceptance suites (V3 §7.1).

These are the measurement instrument for the whole experiment. They are fixed,
versioned, identical across every arm and every run, and are **never** generated
by an agent. Nothing here may import from `governance`, `strategies` or
`agents`: a suite that could see who was being measured would not be a suite.

Each check is a predicate over the artefact:

    passes  <=>  required endpoint implemented  AND  its defect absent

so a suite result is reconstructible by hand from an artefact, which is what
makes the derived confidence in `awareness/confidence.py` auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artefact import Artefact
from .catalogue import DEFAULT_TASK
from .spec import Check, TaskSpec

SUITE_VERSION = DEFAULT_TASK.suite_version

# The default task's checks, kept as module-level names so that callers and tests
# written before tasks were parameterised keep working unchanged.
FUNCTIONAL: tuple[Check, ...] = DEFAULT_TASK.functional
SECURITY: tuple[Check, ...] = DEFAULT_TASK.security


@dataclass(frozen=True)
class SuiteResult:
    name: str
    passed: int
    total: int
    failures: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def green(self) -> bool:
        return self.passed == self.total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": self.total,
            "failures": list(self.failures),
        }


def _run(name: str, checks: tuple[Check, ...], artefact: Artefact) -> SuiteResult:
    failures = tuple(
        c.id
        for c in checks
        if c.unit not in artefact.units or c.defect in artefact.defects
    )
    return SuiteResult(name, len(checks) - len(failures), len(checks), failures)


def functional_suite(artefact: Artefact, spec: TaskSpec | None = None) -> SuiteResult:
    return _run("functional", (spec or DEFAULT_TASK).functional, artefact)


def security_suite(artefact: Artefact, spec: TaskSpec | None = None) -> SuiteResult:
    return _run("security", (spec or DEFAULT_TASK).security, artefact)
