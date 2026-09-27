"""The architectural invariants.

These are not coverage tests. Each one holds a claim the paper makes, so that a
change which quietly breaks the claim fails the build rather than the review.
"""

from __future__ import annotations

import pytest

from agccc.awareness.state import SituationState, SuiteView
from agccc.task.catalogue import REST_AUTH

# These tests exercise the engine against the original task.
UNITS_TOTAL = len(REST_AUTH.units)
from agccc.governance.actions import ALL_ACTIONS, Action
from agccc.governance.decision import DecisionEngine
from agccc.governance.evidence import EvidenceModel
from agccc.governance.policies import EMPTY, default
from agccc.governance.scoring import ScoringConfig, objective


def red_state(**kw) -> SituationState:
    base = dict(
        confidence=0.42,
        functional=SuiteView(8, 8),
        security=SuiteView(1, 7),
        security_failure=True,
        tested_at_least_once=True,
    )
    base.update(kw)
    return SituationState(**base)


# --- filter-then-rank: the invariant Article #2 turns on ------------------


class ScoringSpy:
    """Records every action the ranker is asked to score."""

    def __init__(self) -> None:
        self.seen: list[Action] = []

    def __call__(self, action, state, p_success, endpoints, units_total, cfg):
        self.seen.append(action)
        return objective(action, state, p_success, endpoints, units_total, cfg)


def test_the_ranker_never_sees_an_inadmissible_action():
    """The core of the filter-then-rank invariant (V3 §1).

    Under rank-then-check a prohibited action is merely outweighed, and anything
    outweighed can be outweighed back. Under filter-then-rank it is structurally
    unreachable — the scorer is never even called on it.
    """
    spy = ScoringSpy()
    engine = DecisionEngine(policies=default(), scorer=spy)
    state = red_state()

    decision = engine.decide(state, EvidenceModel(), endpoints_implemented=5,
                             units_total=UNITS_TOTAL)

    assert Action.ACCEPT in decision.rejected, "policy must prohibit release while red"
    assert Action.ACCEPT not in spy.seen, "a rejected action must never be scored"
    assert Action.ACCEPT not in decision.scores
    assert set(spy.seen).isdisjoint(decision.rejected)


def test_no_utility_however_large_can_make_a_prohibited_action_selectable():
    """Structural unreachability, stated as a test.

    A scorer that returns +inf for ACCEPT and -inf for everything else still
    cannot cause ACCEPT to be selected, because ACCEPT never reaches the ranker.
    The same scorer trivially selects ACCEPT once governance is removed, which
    is what makes this a property of the *ordering* and not of the numbers.
    """

    def perverse(action, state, p_success, endpoints, units_total, cfg):
        return float("inf") if action is Action.ACCEPT else float("-inf")

    state = red_state()

    governed = DecisionEngine(policies=default(), scorer=perverse)
    assert governed.decide(state, EvidenceModel(), 5, UNITS_TOTAL).selected is not Action.ACCEPT

    ungoverned = DecisionEngine(policies=EMPTY, scorer=perverse)
    assert ungoverned.decide(state, EvidenceModel(), 5, UNITS_TOTAL).selected is Action.ACCEPT


def test_escalation_when_nothing_is_admissible():
    """No candidate survives -> escalate. Never relax a policy to keep going."""

    class ProhibitEverything:
        def prohibited(self, state):
            return {a: "TEST_ALL" for a in ALL_ACTIONS}

    engine = DecisionEngine(policies=ProhibitEverything())
    d = engine.decide(red_state(), EvidenceModel(), 5, UNITS_TOTAL)
    assert d.escalated and d.selected is None


# --- evidence: behaviour change without a rule ---------------------------


def test_p_success_decays_on_the_documented_schedule():
    """V3 §9.3 — 0.500 -> 0.333 -> 0.250 -> 0.200 under repeated failure."""
    ev = EvidenceModel()
    observed = [ev.p_success(Action.RETRY_CODER)]
    for _ in range(3):
        ev.record(Action.RETRY_CODER, False)
        observed.append(ev.p_success(Action.RETRY_CODER))
    assert observed == pytest.approx([0.5, 1 / 3, 0.25, 0.2])


def test_repeated_failure_flips_the_selection_with_no_rule_and_no_counter():
    """The paper's central mechanism.

    Retrying the Coder starts ahead and ends behind, purely because its
    posterior decayed while an untried alternative held its prior. There is no
    threshold and no retry limit anywhere in the code path.
    """
    # Functional worse than security, so RETRY_CODER genuinely starts ahead.
    state = SituationState(
        confidence=0.55,
        functional=SuiteView(2, 8),
        security=SuiteView(5, 7),
        functional_failure=True,
        security_failure=True,
        tested_at_least_once=True,
    )
    engine = DecisionEngine(policies=default())
    ev = EvidenceModel()

    first = engine.decide(state, ev, 5, UNITS_TOTAL).selected
    assert first is Action.RETRY_CODER

    for _ in range(3):
        ev.record(Action.RETRY_CODER, False)

    later = engine.decide(state, ev, 5, UNITS_TOTAL).selected
    assert later is not Action.RETRY_CODER, "decision should move off a failing action"


def test_no_hard_coded_retry_limit_exists_in_the_control_path():
    """Raw_planning §12 Guideline 6, enforced rather than trusted.

    Scope matters here. The check covers the *control path* — situation
    awareness, governance, the adaptive strategy and the kernel — and not
    `agents/sim`, which is the simulated world rather than the architecture, nor
    the baselines, which are *supposed* to escalate on a fixed constant. That is
    B2's defining weakness and the thing AGCCC is being compared against.

    Comments and docstrings are stripped before matching, so prose describing
    the prohibition does not trip the check on itself.
    """
    import ast
    import pathlib
    import re

    from tests.helpers import code_only

    root = pathlib.Path(__file__).resolve().parents[1] / "agccc"
    control_path = ("awareness", "governance", "kernel")
    banned = re.compile(r"(retry_count|retries?\s*[<>=]=?\s*\d|attempts?\b[^=\n]*[<>]=?\s*\d)")

    offenders = []
    targets = [p for d in control_path for p in (root / d).rglob("*.py")]
    targets.append(root / "strategies" / "adaptive.py")
    for path in targets:
        for i, line in enumerate(code_only(path).splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, "hard-coded retry logic found:\n" + "\n".join(offenders)


# --- policies ------------------------------------------------------------


def test_policies_load_from_yaml_not_python():
    ps = default()
    assert ps.version and len(ps.policies) >= 3
    assert any(p.id == "SECURITY_DEPLOYMENT_BLOCK" for p in ps.policies)


def test_release_is_prohibited_while_either_suite_is_red():
    ps = default()
    assert Action.ACCEPT in ps.prohibited(red_state())
    functional_red = SituationState(
        functional=SuiteView(6, 8), security=SuiteView(7, 7),
        functional_failure=True, tested_at_least_once=True, confidence=0.875,
    )
    assert Action.ACCEPT in ps.prohibited(functional_red)


def test_release_is_permitted_once_everything_is_green():
    green = SituationState(
        confidence=1.0,
        functional=SuiteView(8, 8),
        security=SuiteView(7, 7),
        tested_at_least_once=True,
        deployment_allowed=True,
    )
    assert Action.ACCEPT not in default().prohibited(green)


def test_untested_artefact_cannot_be_released():
    fresh = SituationState(tested_at_least_once=False)
    assert Action.ACCEPT in default().prohibited(fresh)


# --- scoring config ------------------------------------------------------

def test_scoring_hash_changes_with_configuration():
    assert ScoringConfig().hash() != ScoringConfig(lambda_risk=0.1).hash()
    assert ScoringConfig().hash() == ScoringConfig().hash()
