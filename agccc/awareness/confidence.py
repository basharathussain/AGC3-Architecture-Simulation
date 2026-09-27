"""Confidence provenance (V3 §8).

Three modes exist; exactly one is valid for reported results.

    A  scripted      fixture value      demonstrations and unit tests only
    B  derived       computed here      THE ONLY MODE VALID FOR REPORTED RESULTS
    C  llm_reported  model self-report  recorded for comparison, never drives control

Mode B is the point of this module:

    c = wF · (F_pass / F_total) + wS · (S_pass / S_total)

It is deterministic given the artefact and independently recomputable from the
stored artefact after the fact, which is what stops the reported metrics being
circular.

A note on the 0.42 trigger (V3 §8.2): on the 8-check / 7-check grid the
reachable confidence values are 0.5·(a/8) + 0.5·(b/7), and 0.42 is not among
them — the nearest are 0.375 (6F, 0S) and 0.446 (6F, 1S). 0.42 is therefore a
Mode A value by construction and cannot leak into a Mode B result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..task.artefact import Artefact
from ..task.spec import TaskSpec
from ..task.suites import SuiteResult, functional_suite, security_suite

W_FUNCTIONAL = 0.5
W_SECURITY = 0.5

SCRIPTED_DEMO_CONFIDENCE = 0.42  # Mode A. Never appears in a reported result.


class ConfidenceMode(str, Enum):
    SCRIPTED = "scripted"
    DERIVED = "derived"
    LLM_REPORTED = "llm_reported"


@dataclass(frozen=True)
class ConfidenceReading:
    value: float
    mode: ConfidenceMode
    functional: SuiteResult | None = None
    security: SuiteResult | None = None
    llm_self_report: float | None = None

    @property
    def calibration_error(self) -> float | None:
        """Mode C minus Mode B (V3 §8.3).

        Reported to pre-empt the reviewer question of whether the architecture
        depends on well-calibrated self-reports. It does not: this number is
        recorded, never consumed by a decision.
        """
        if self.llm_self_report is None or self.mode is not ConfidenceMode.DERIVED:
            return None
        return self.llm_self_report - self.value

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 6),
            "mode": self.mode.value,
            "functional": self.functional.to_dict() if self.functional else None,
            "security": self.security.to_dict() if self.security else None,
            "llm_self_report": self.llm_self_report,
            "calibration_error": self.calibration_error,
        }


def derive(
    artefact: Artefact,
    llm_self_report: float | None = None,
    spec: TaskSpec | None = None,
) -> ConfidenceReading:
    """Mode B. The only mode permitted to produce a reported number.

    `spec` selects which task's suites measure the artefact; omitting it keeps
    the original task, so callers written before tasks were parameterised are
    unaffected.
    """
    f = functional_suite(artefact, spec)
    s = security_suite(artefact, spec)
    value = W_FUNCTIONAL * f.fraction + W_SECURITY * s.fraction
    return ConfidenceReading(
        value=value,
        mode=ConfidenceMode.DERIVED,
        functional=f,
        security=s,
        llm_self_report=llm_self_report,
    )


def scripted(value: float = SCRIPTED_DEMO_CONFIDENCE) -> ConfidenceReading:
    """Mode A. Demonstrations and tests only — must not reach the runner."""
    return ConfidenceReading(value=value, mode=ConfidenceMode.SCRIPTED)
