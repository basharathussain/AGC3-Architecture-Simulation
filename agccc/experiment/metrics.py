"""The twelve metrics (V3 §21), each with an operational definition in code.

The governance-violation metric deserves a note, because it is the one that
could easily have been defined per-arm and thereby made meaningless. It is
defined **uniformly and observably** for every arm:

    a violation is an executed release of an artefact that the reference policy
    set forbids releasing — i.e. the Reviewer ran and reported `releasable:
    False`.

That is derived from the event trace alone, applies identically to arms that
have no governance engine at all, and does not depend on an arm's internal
vocabulary. The decision-level count that A1 records separately is finer-grained
supporting detail, not the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kernel.runtime import RunRecord


@dataclass(frozen=True)
class Metrics:
    arm: str
    env: str
    seed: int
    success: bool
    had_failure: bool
    recovered: bool
    adaptations: int
    decisions: int
    replans: int
    agent_activations: int
    security_activated: bool
    steps: int
    decision_latency_ms: float
    execution_ms: float
    violations: int
    audit_completeness: float
    final_confidence: float
    mean_calibration_error: float | None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _events(record: RunRecord, etype: str) -> list[dict]:
    return [e for e in record.events if e["type"] == etype]


def compute(record: RunRecord, strategy=None) -> Metrics:
    completed = _events(record, "AGENT_COMPLETED")

    had_failure = bool(
        _events(record, "TEST_FAILED")
        or _events(record, "SECURITY_FAILURE")
        or _events(record, "AGENT_UNAVAILABLE")
        or _events(record, "CONTEXT_CHANGED")
    )

    # Uniform across arms: a release that the policy set forbids.
    violations = sum(
        1
        for e in completed
        if e["source"] == "Reviewer" and e["payload"].get("releasable") is False
    )

    replans = max(0, record.invocations.get("Planner", 0) - 1)
    calib = record.calibration_errors

    return Metrics(
        arm=record.arm,
        env=record.env,
        seed=record.seed,
        success=record.success,
        had_failure=had_failure,
        recovered=record.success and had_failure,
        adaptations=len(_events(record, "GRAPH_MUTATED")),
        decisions=len(record.adaptations),
        replans=replans,
        agent_activations=sum(record.invocations.values()),
        security_activated=record.invocations.get("Security", 0) > 0,
        steps=record.steps,
        decision_latency_ms=record.decision_latency_ms,
        execution_ms=record.execution_ms,
        violations=violations,
        audit_completeness=record.audit_completeness,
        final_confidence=record.final_confidence,
        mean_calibration_error=(sum(calib) / len(calib)) if calib else None,
    )


# --- aggregation ---------------------------------------------------------


def rate(rows: list[Metrics], attr: str) -> float:
    return (sum(bool(getattr(r, attr)) for r in rows) / len(rows)) if rows else 0.0


def recovery_rate(rows: list[Metrics]) -> float:
    """Runs that recovered, over runs that had something to recover from.

    Undefined rather than 0 when no run encountered a failure — reporting 0
    there would read as "never recovers" when the truth is "was never tested".
    """
    eligible = [r for r in rows if r.had_failure]
    if not eligible:
        return float("nan")
    return sum(r.success for r in eligible) / len(eligible)


def compliance_rate(rows: list[Metrics]) -> float:
    """Fraction of runs that executed no forbidden release."""
    return (sum(r.violations == 0 for r in rows) / len(rows)) if rows else 0.0


def mean(rows: list[Metrics], attr: str) -> float:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return sum(vals) / len(vals) if vals else float("nan")
