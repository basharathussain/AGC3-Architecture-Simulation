"""Graph operations, layer boundaries, and experimental comparability."""

from __future__ import annotations

import pathlib
import re

import pytest

from agccc.agents.base import CODER, PLANNER, REVIEWER, SECURITY, TESTER
from agccc.agents.sim.roles import build_roster
from agccc.agents.sim.world import World, WorldParams
from agccc.experiment import arms as A
from agccc.experiment.injection import Injector, for_env
from agccc.experiment.runner import Experiment
from agccc.graph import ops
from agccc.graph.diff import diff
from agccc.graph.model import chain
from agccc.kernel.runtime import Runtime

ROOT = pathlib.Path(__file__).resolve().parents[1] / "agccc"


# --- graph ---------------------------------------------------------------


def test_insert_after_splices_rather_than_appends():
    g = chain(PLANNER, CODER, TESTER, REVIEWER)
    g2 = ops.insert_after(g, TESTER, SECURITY)
    assert (TESTER, SECURITY) in g2.edges
    assert (SECURITY, REVIEWER) in g2.edges
    assert (TESTER, REVIEWER) not in g2.edges, "the spliced node must interpose"
    assert g2.order == (PLANNER, CODER, TESTER, SECURITY, REVIEWER)


def test_operations_never_mutate_the_input_graph():
    g = chain(PLANNER, CODER, TESTER)
    before = g.digest()
    ops.add_node(g, SECURITY)
    ops.add_edge(g, TESTER, CODER)
    ops.insert_after(g, CODER, SECURITY)
    assert g.digest() == before


def test_graph_diff_reports_the_structural_change():
    g = chain(PLANNER, CODER, TESTER, REVIEWER)
    g2 = ops.add_edge(ops.add_node(g, SECURITY), TESTER, SECURITY)
    d = diff(g, g2)
    assert d.nodes_added == (SECURITY,)
    assert d.edges_added == ((TESTER, SECURITY),)
    assert d.magnitude == 2 and not d.is_empty


def test_identical_graphs_diff_to_nothing():
    g = chain(PLANNER, CODER)
    assert diff(g, g).is_empty


def test_entry_node_cannot_be_removed():
    g = chain(PLANNER, CODER)
    with pytest.raises(ValueError):
        ops.remove_node(g, PLANNER)


def test_edges_must_reference_known_nodes():
    with pytest.raises(ValueError):
        chain(PLANNER, CODER).__class__(
            nodes=frozenset({PLANNER}), edges=frozenset({(PLANNER, "Ghost")}), entry=PLANNER
        )


# --- layer boundaries ----------------------------------------------------


def test_awareness_does_not_import_governance_or_strategies():
    """Situation awareness answers "what is happening", never "what next".

    Enforced by import direction: if the observer could reach the governance
    engine, the separation would be a convention rather than an architecture.
    """
    for path in (ROOT / "awareness").rglob("*.py"):
        src = path.read_text()
        for forbidden in ("governance", "strategies", "graph"):
            assert not re.search(rf"^\s*from .*{forbidden}", src, re.M), (
                f"{path.name} imports {forbidden}"
            )


def test_agents_cannot_reach_the_graph_or_the_policies():
    """Raw_planning §12 Guideline 5 — agents report, they do not govern."""
    for path in (ROOT / "agents").rglob("*.py"):
        src = path.read_text()
        for forbidden in ("governance", "strategies"):
            assert not re.search(rf"^\s*from .*{forbidden}", src, re.M), (
                f"{path.name} imports {forbidden}"
            )


def test_the_task_suites_cannot_see_who_is_being_measured():
    for path in (ROOT / "task").rglob("*.py"):
        src = path.read_text()
        for forbidden in ("governance", "strategies", "agents", "kernel"):
            assert not re.search(rf"^\s*from .*{forbidden}", src, re.M)


def test_the_kernel_contains_no_orchestration_logic():
    """The microkernel must be small and boring (Raw_planning §2).

    It may not name a single agent, policy or retry rule in *code* — the
    docstring is allowed to explain that it contains none of them, which is why
    comments and docstrings are stripped before the check.
    """
    from tests.helpers import code_only

    src = code_only(ROOT / "kernel" / "runtime.py")
    for leak in ("Security", "Planner", "Coder", "Tester", "retry", "policy", "Policy"):
        assert leak not in src, f"kernel/runtime.py contains {leak!r} in code"


# --- experimental comparability ------------------------------------------


def test_every_arm_faces_an_identical_world_for_a_given_seed():
    """Mitigation #1 — one draw, shared across arms."""
    a, b = World(7), World(7)
    assert a.initial_defects() == b.initial_defects()
    assert all(a.coder_clears(k) == b.coder_clears(k) for k in range(6))
    assert all(a.specialist_clears(k) == b.specialist_clears(k) for k in range(6))


def test_world_draws_are_index_stable_not_sequence_dependent():
    """Arms consume randomness at different rates and must not desynchronise.

    Reading attempt 5 first must give the same answer as reading it last.
    """
    w = World(11)
    late_first = w.coder_clears(5)
    for k in range(5):
        w.coder_clears(k)
    assert w.coder_clears(5) == late_first


def test_different_seeds_produce_different_worlds():
    assert World(1).initial_defects() != World(2).initial_defects()


def test_injection_schedule_depends_only_on_the_seed():
    a, b = Injector(4), Injector(4)
    assert a.to_dict() == b.to_dict()
    assert all(a.unavailable_at(s) == b.unavailable_at(s) for s in range(20))
    assert Injector(4).to_dict() != Injector(5).to_dict()


def test_nominal_environment_injects_nothing():
    inj = for_env("nominal", 3)
    assert all(inj.unavailable_at(s) == frozenset() for s in range(20))


def test_runs_are_reproducible():
    exp = Experiment(n=3)
    first = exp.run_one("A2", "perturbed", 2)[0].to_dict()
    second = exp.run_one("A2", "perturbed", 2)[0].to_dict()
    for key in ("final_artefact", "functional", "security", "outcome", "steps"):
        assert first[key] == second[key]


def test_adaptive_arm_actually_mutates_the_graph():
    """G_{t+1} != G_t must genuinely occur, not merely be possible."""
    w = World(1)
    strategy = A.build("A2")
    rec = Runtime(build_roster(w), strategy).run("A2", "nominal", 1)
    assert len(rec.graph_history) > 1
    assert any(e["type"] == "GRAPH_MUTATED" for e in rec.events)


def test_non_adaptive_arms_never_mutate_the_graph():
    w = World(1)
    for arm in ("B1", "B2", "B3"):
        rec = Runtime(build_roster(w), A.build(arm)).run(arm, "nominal", 1)
        assert len(rec.graph_history) == 1, f"{arm} changed its topology"


def test_supervisor_can_reach_the_security_specialist():
    """Fairness: B2 must not lose by being denied the specialist.

    Without this the comparison would be definitional rather than empirical.
    """
    from agccc.strategies.supervisor import ROSTER

    assert SECURITY in ROSTER
    g = A.build("B2").initial_graph()
    assert SECURITY in g.nodes
    assert (TESTER, SECURITY) in g.edges


def test_a1_and_a2_differ_only_in_the_policy_set():
    a1, a2 = A.build("A1"), A.build("A2")
    assert a1.engine.scoring == a2.engine.scoring
    assert a1.engine.action_space == a2.engine.action_space
    assert len(a1.engine.policies.policies) == 0
    assert len(a2.engine.policies.policies) > 0


def test_audit_record_is_complete_for_every_adaptation():
    """Auditability metric — the record must carry every field the paper names."""
    rec = Runtime(build_roster(World(3)), A.build("A2")).run("A2", "nominal", 3)
    assert rec.adaptations, "expected at least one governed decision"
    assert rec.audit_completeness == 1.0
    first = rec.adaptations[0]
    for field in ("trigger", "candidates", "selected", "graph_before", "graph_after", "state_after"):
        assert first[field], f"audit record missing {field}"


def test_confidence_is_derived_mode_b_in_every_reported_run():
    """V3 §8 — the scripted 0.42 trigger must never reach a reported result."""
    exp = Experiment(n=5)
    rows, records, repro = exp.run_all()
    assert repro["confidence_mode"] == "derived (Mode B)"
    reachable = {round(0.5 * (a / 8) + 0.5 * (b / 7), 6) for a in range(9) for b in range(8)}
    for r in records:
        assert round(r["final_confidence"], 6) in reachable
