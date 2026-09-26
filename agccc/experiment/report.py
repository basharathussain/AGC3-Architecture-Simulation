"""Produce every artefact the paper needs, in one command.

    python -m agccc.experiment.report

Writes into `results/`:

    table3.tex          Table III   main comparison
    table4.tex          Table IV    ablation
    stats.tex                       Mann-Whitney U / Cliff's delta / Holm
    e1_trace.tex                    E1 candidate-score trace
    sensitivity.tex                 lambda_R and world-parameter sweeps
    fig2.pdf                        completion and recovery
    fig3.pdf                        governance compliance and the soft/hard sweep
    metrics.json, runs.json, reproducibility.json
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agents.sim.world import WorldParams
from ..governance.scoring import ScoringConfig
from . import emit
from . import metrics as M
from .runner import RESULTS_DIR, Experiment

LAMBDA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
COMPETENCE_GRID = ((0.05, 0.95), (0.15, 0.90), (0.30, 0.75), (0.45, 0.60))

# The utility-misspecification family (E5).
#
# Each entry is a scoring configuration a competent engineer might plausibly
# author without calibration data: a risk weight anywhere from "ignore risk" to
# "risk dominates", a cost weight spanning cheap-to-expensive adaptation, and a
# release-risk constant that is either severe or merely serious. The family is
# declared here, in full, and every member is reported — the headline is the
# *worst* case per arm, not a selected member.
#
# The point of the sweep is that a soft penalty is a claim about the utility
# function you wrote, whereas a hard constraint is a claim about the ones you
# did not. Only a family can distinguish those.
MISSPEC_LAMBDA_RISK = (0.0, 0.1, 0.2, 0.3, 0.5)
MISSPEC_LAMBDA_COST = (0.05, 0.15, 0.30)
MISSPEC_RELEASE_RISK = (0.6, 1.0)


def lambda_sweep(n: int) -> list[dict]:
    """Mitigation #2: is the conclusion an artefact of one parameter setting?

    The headline uses lambda_R = 0.5, the *strongest* soft penalty and therefore
    the setting most favourable to the ungoverned arm. If the governed arm still
    differs there, the finding is not manufactured by a weak comparator.
    """
    out = []
    for lam in LAMBDA_GRID:
        exp = Experiment(n=n, scoring=ScoringConfig(lambda_risk=lam))
        row = {"lambda_risk": lam}
        for arm in ("A1", "A2"):
            succ = viol = 0
            for seed in range(1, n + 1):
                record, strategy = exp.run_one(arm, "nominal", seed)
                m = M.compute(record, strategy)
                succ += m.success
                viol += len(getattr(strategy, "violations", []))
            row[f"{arm}_success"] = succ / n
            row[f"{arm}_violations"] = viol
        out.append(row)
    return out


def competence_sweep(n: int) -> list[dict]:
    """Does the conclusion survive a different simulated world?

    Sweeps the two agent-competence parameters that decide whether a red
    security suite is escapable by retrying the Coder. If AGCCC's advantage
    disappeared as the Coder got better at security, that would be a real
    boundary condition and it belongs in the paper.
    """
    out = []
    for p_coder, p_spec in COMPETENCE_GRID:
        params = WorldParams(
            p_fix_security_coder=p_coder, p_fix_security_specialist=p_spec
        )
        exp = Experiment(n=n, world_params=params)
        row = {"p_fix_security_coder": p_coder, "p_fix_security_specialist": p_spec}
        for arm in ("B2", "A2"):
            succ = 0
            for seed in range(1, n + 1):
                record, _ = exp.run_one(arm, "nominal", seed)
                succ += M.compute(record).success
            row[f"{arm}_success"] = succ / n
        out.append(row)
    return out


def misspecification_sweep(n: int) -> list[dict]:
    """E5 — robustness of compliance to utility misspecification.

    The governed and ungoverned arms are indistinguishable at the declared
    scoring configuration, and this is the experiment that says why that is the
    finding rather than a non-result. A safety mechanism is not evaluated on its
    mean: it is evaluated on whether it holds across the configurations you did
    not anticipate.

    Both arms see identical seeds and identical scoring within each
    configuration, so the pairing is exact and the comparison is made with a
    signed-rank test rather than an independent-samples one.
    """
    out = []
    for lr in MISSPEC_LAMBDA_RISK:
        for lc in MISSPEC_LAMBDA_COST:
            for rr in MISSPEC_RELEASE_RISK:
                cfg = ScoringConfig(
                    lambda_risk=lr, lambda_cost=lc, risk_accept_security_red=rr
                )
                exp = Experiment(n=n, scoring=cfg)
                row = {
                    "lambda_risk": lr,
                    "lambda_cost": lc,
                    "release_risk": rr,
                    "scoring_hash": cfg.hash(),
                }
                for arm in ("A1", "A2"):
                    runs = [
                        M.compute(exp.run_one(arm, "nominal", s)[0])
                        for s in range(1, n + 1)
                    ]
                    row[f"{arm}_compliance"] = M.compliance_rate(runs)
                    row[f"{arm}_success"] = M.rate(runs, "success")
                    row[f"{arm}_violations"] = sum(r.violations for r in runs)
                out.append(row)
    return out


PRESSURE_POINTS = 21  # lambda_R from 0.5 down to 0.0


def pressure_sweep(n: int, points: int = PRESSURE_POINTS) -> list[dict]:
    """E6 — governance robustness as utility pressure rises.

    "Utility pressure" is not a new mechanism; it is the declared risk weight
    read backwards. As lambda_R falls, the penalty on releasing a non-green
    artefact shrinks, so the prohibited action becomes relatively more
    attractive to the ranker. Pressure is reported normalised,
    `1 - lambda_R / 0.5`, so that 0 is the declared configuration and 1 is a
    scorer that ignores release risk entirely.

    The prediction is asymmetric and worth stating before looking: the
    ungoverned arm should fail progressively, because its safety is a property
    of the weighting. The governed arm should not move at all, because a
    filtered action is unreachable at every weighting. A flat line here is the
    result, not an absence of one.
    """
    lam_max = ScoringConfig().lambda_risk
    out = []
    for i in range(points):
        lam = lam_max * (1 - i / (points - 1))
        exp = Experiment(n=n, scoring=ScoringConfig(lambda_risk=lam))
        row = {"lambda_risk": lam, "pressure": 1 - lam / lam_max}
        for arm in ("A1", "A2"):
            rows = [M.compute(exp.run_one(arm, "nominal", s)[0]) for s in range(1, n + 1)]
            row[f"{arm}_violation_rate"] = 1 - M.compliance_rate(rows)
            row[f"{arm}_success"] = M.rate(rows, "success")
        out.append(row)
    return out


def main(n: int = 30, out_dir: Path = RESULTS_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("main experiment ...")
    exp = Experiment(n=n)
    rows, records, repro = exp.run_all(verbose=True)

    print("lambda_R sweep ...")
    sweep = lambda_sweep(n)
    print("competence sweep ...")
    grid = competence_sweep(n)

    repro["lambda_sweep"] = sweep
    repro["competence_sweep"] = grid

    (out_dir / "metrics.json").write_text(json.dumps([r.to_dict() for r in rows], indent=1))
    (out_dir / "runs.json").write_text(json.dumps(records, indent=1))
    (out_dir / "reproducibility.json").write_text(json.dumps(repro, indent=2))

    (out_dir / "table3.tex").write_text(emit.table3(rows))
    (out_dir / "table4.tex").write_text(emit.table4(rows))

    print("misspecification sweep (E5) ...")
    misspec = misspecification_sweep(n)
    repro["misspecification_sweep"] = misspec
    misspec_tex, misspec_comp = emit.misspecification(misspec)
    (out_dir / "misspecification.tex").write_text(misspec_tex)

    # The paired misspecification test joins the same Holm family, so the
    # correction covers every hypothesis tested rather than a convenient subset.
    stats_tex, comps = emit.significance(rows, extra=[misspec_comp])
    (out_dir / "stats.tex").write_text(stats_tex)

    e1_tex, e1_run = emit.e1_trace(records)
    (out_dir / "e1_trace.tex").write_text(e1_tex)

    (out_dir / "sensitivity.tex").write_text(emit.sensitivity_table(sweep, grid))

    print("supplementary power check (n=100) ...")
    rows_100, _, _ = Experiment(n=100).run_all()
    (out_dir / "power.tex").write_text(emit.power_note(rows, rows_100))
    emit.figures(rows, sweep, out_dir, misspec=misspec)

    print("pressure sweep (E6) ...")
    pressure = pressure_sweep(n)
    repro["pressure_sweep"] = pressure
    repro["robustness_figure"] = emit.figure_robustness(pressure, out_dir)

    # Rewritten last: the sweeps above add to `repro` after the first write, and
    # a reproducibility record missing an experiment it reports would be worse
    # than none at all.
    (out_dir / "reproducibility.json").write_text(json.dumps(repro, indent=2))

    print("\n--- significance ---")
    for c in comps:
        mark = "*" if c.significant else " "
        print(f" {mark} {c.label:44s} p_holm={c.p_holm:.4f} delta={c.delta:+.3f} ({c.magnitude})")
    if e1_run:
        print(f"\nE1 trace: seed {e1_run['seed']}, {len(e1_run['decisions_trace'])} rounds")

    print(f"\nwrote {len(list(out_dir.glob('*.tex')))} .tex and "
          f"{len(list(out_dir.glob('*.pdf')))} .pdf into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

