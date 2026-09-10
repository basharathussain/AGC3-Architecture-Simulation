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

    stats_tex, comps = emit.significance(rows)
    (out_dir / "stats.tex").write_text(stats_tex)

    e1_tex, e1_run = emit.e1_trace(records)
    (out_dir / "e1_trace.tex").write_text(e1_tex)

    (out_dir / "sensitivity.tex").write_text(emit.sensitivity_table(sweep, grid))

    print("supplementary power check (n=100) ...")
    rows_100, _, _ = Experiment(n=100).run_all()
    (out_dir / "power.tex").write_text(emit.power_note(rows, rows_100))
    emit.figures(rows, sweep, out_dir)

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
