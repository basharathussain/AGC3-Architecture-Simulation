"""P0 — the measurement instrument, against hand-computed fixtures.

Every expected number below is worked out by hand in the comment beside it. If
these are wrong, nothing downstream means anything.
"""

from __future__ import annotations

import pytest

from agccc.awareness.confidence import (
    SCRIPTED_DEMO_CONFIDENCE,
    ConfidenceMode,
    derive,
    scripted,
)
from agccc.task.artefact import EMPTY, Artefact
from agccc.task.defects import (
    ALL_ENDPOINTS,
    FUNCTIONAL_DEFECTS,
    SECURITY_DEFECTS,
    Defect,
    Endpoint,
)
from agccc.task.suites import FUNCTIONAL, SECURITY, functional_suite, security_suite


def perfect() -> Artefact:
    """All five endpoints, no defects."""
    return Artefact(endpoints=ALL_ENDPOINTS, defects=frozenset())


# --- catalogue integrity -------------------------------------------------


def test_every_check_has_exactly_one_defect_and_vice_versa():
    checked = {c.defect for c in FUNCTIONAL + SECURITY}
    assert checked == set(Defect), "a defect with no check is unobservable"
    assert len(FUNCTIONAL) == 8 and len(SECURITY) == 7
    assert {c.defect for c in FUNCTIONAL} == set(FUNCTIONAL_DEFECTS)
    assert {c.defect for c in SECURITY} == set(SECURITY_DEFECTS)


def test_check_ids_are_unique():
    ids = [c.id for c in FUNCTIONAL + SECURITY]
    assert len(ids) == len(set(ids))


# --- suites --------------------------------------------------------------


def test_empty_artefact_fails_everything():
    # No endpoints implemented, so all 8 + 7 checks fail on the endpoint clause.
    f, s = functional_suite(EMPTY), security_suite(EMPTY)
    assert (f.passed, f.total) == (0, 8)
    assert (s.passed, s.total) == (0, 7)
    assert not f.green and not s.green


def test_perfect_artefact_passes_everything():
    f, s = functional_suite(perfect()), security_suite(perfect())
    assert (f.passed, f.total) == (8, 8)
    assert (s.passed, s.total) == (7, 7)
    assert f.green and s.green


def test_single_defect_fails_exactly_one_check():
    a = perfect().with_defects({Defect.ALG_NONE_ACCEPTED})
    f, s = functional_suite(a), security_suite(a)
    assert (f.passed, f.total) == (8, 8), "a security defect must not move the functional suite"
    assert (s.passed, s.total) == (6, 7)
    assert s.failures == ("S4",)


def test_missing_endpoint_fails_all_its_checks_in_both_suites():
    # Dropping GET /me kills F5, F6 (functional) and S3, S4, S5 (security).
    a = Artefact(endpoints=ALL_ENDPOINTS - {Endpoint.ME})
    f, s = functional_suite(a), security_suite(a)
    assert f.failures == ("F5", "F6")
    assert s.failures == ("S3", "S4", "S5")


def test_suites_are_read_only():
    a = perfect().with_defects({Defect.SQLI_LOGIN})
    before = a.to_dict()
    functional_suite(a)
    security_suite(a)
    assert a.to_dict() == before


# --- derived confidence (Mode B) -----------------------------------------


def test_confidence_perfect_is_one():
    assert derive(perfect()).value == pytest.approx(1.0)


def test_confidence_empty_is_zero():
    assert derive(EMPTY).value == pytest.approx(0.0)


def test_confidence_hand_computed_green_functional_red_security():
    """The paper's worked example: functional passing, security largely failing.

    All 8 functional pass; 6 of 7 security defects present, so 1 security check
    passes.  c = 0.5·(8/8) + 0.5·(1/7) = 0.5 + 0.0714285... = 0.5714285...
    """
    a = perfect().with_defects(set(SECURITY_DEFECTS[:6]))
    c = derive(a)
    assert (c.functional.passed, c.security.passed) == (8, 1)
    assert c.value == pytest.approx(0.5 + 1 / 14)
    assert c.value == pytest.approx(0.5714285714, abs=1e-9)


def test_confidence_hand_computed_mixed():
    """6 of 8 functional and 3 of 7 security passing.

    Remove 2 functional defects' worth of passes and 4 security:
    c = 0.5·(6/8) + 0.5·(3/7) = 0.375 + 0.2142857... = 0.5892857...
    """
    a = perfect().with_defects(set(FUNCTIONAL_DEFECTS[:2]) | set(SECURITY_DEFECTS[:4]))
    c = derive(a)
    assert (c.functional.passed, c.security.passed) == (6, 3)
    assert c.value == pytest.approx(0.375 + 3 / 14)
    assert c.value == pytest.approx(0.5892857143, abs=1e-9)


def test_confidence_is_bounded_and_deterministic():
    a = perfect().with_defects({Defect.SQLI_LOGIN, Defect.ME_BROKEN})
    first = derive(a).value
    assert 0.0 <= first <= 1.0
    assert derive(a).value == first


def test_derive_is_mode_b_and_scripted_is_mode_a():
    assert derive(perfect()).mode is ConfidenceMode.DERIVED
    assert scripted().mode is ConfidenceMode.SCRIPTED
    assert scripted().value == SCRIPTED_DEMO_CONFIDENCE


def test_the_scripted_trigger_is_unreachable_under_mode_b():
    """V3 §8.2 — 0.42 must be a Mode A value that cannot leak into a result.

    Enumerate the entire reachable grid and confirm 0.42 is not on it.
    """
    reachable = {round(0.5 * (a / 8) + 0.5 * (b / 7), 10) for a in range(9) for b in range(8)}
    assert round(SCRIPTED_DEMO_CONFIDENCE, 10) not in reachable


def test_calibration_error_records_but_does_not_alter_the_derived_value():
    a = perfect().with_defects(set(SECURITY_DEFECTS[:6]))
    honest = derive(a)
    boastful = derive(a, llm_self_report=0.95)
    assert boastful.value == honest.value, "a self-report must never move Mode B"
    assert boastful.calibration_error == pytest.approx(0.95 - honest.value)
    assert honest.calibration_error is None


def test_scripted_reading_has_no_calibration_error():
    assert scripted().calibration_error is None


# --- artefact ------------------------------------------------------------


def test_artefact_is_immutable_and_operations_return_copies():
    a = EMPTY
    b = a.with_defects({Defect.SQLI_LOGIN})
    assert a.defects == frozenset() and b.defects == {Defect.SQLI_LOGIN}
    assert a.digest() != b.digest()
    c = b.without_defects({Defect.SQLI_LOGIN})
    assert c.digest() == a.digest(), "digest must depend only on content"


def test_is_complete():
    assert perfect().is_complete
    assert not EMPTY.is_complete
