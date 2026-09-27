"""E7 — replication of the primary comparison across task specifications.

The study's primary campaign is one task (`rest_auth`), pre-specified, with the
family of seven comparisons corrected together. This module adds two further
authored tasks and re-runs the same comparison on each.

It is a **robustness check, not a sample of the task space.** Three authored
specifications do not estimate between-task variance and nothing here is
corrected into the primary family; the two additional tasks were written after
the primary campaign had been analysed, so they are reported as exploratory in
exactly the way the misspecification family is. What they can settle is narrower
and still worth settling: whether the ordering between arms is an artefact of
the one defect profile that was written first.

Regenerate with:  python -m agccc.experiment.tasks_report
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

from ..task.catalogue import TASKS
from .report import RESULTS_DIR, misspecification_sweep
from .runner import Experiment

ARMS = ("B1", "B2", "B3", "A1", "A2")


def _mcnemar(a: list[bool], b: list[bool]) -> tuple[float, int, int]:
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    d = n10 + n01
    p = 1.0 if d == 0 else float(binomtest(n10, d, 0.5, alternative="two-sided").pvalue)
    return p, n10, n01


def campaign(n: int = 30) -> dict:
    out: dict = {}
    for key, spec in TASKS.items():
        rows, _, _ = Experiment(n=n, task=spec).run_all()
        rs = [r.to_dict() if hasattr(r, "to_dict") else r for r in rows]
        by = {(r["arm"], r["env"], r["seed"]): r for r in rs}
        seeds = sorted({r["seed"] for r in rs})
        ov = spec.world_overrides or {}
        entry: dict = {
            "name": spec.name,
            "units": len(spec.units),
            "functional_checks": len(spec.functional),
            "security_checks": len(spec.security),
            "competence_gap": round(
                ov.get("p_fix_security_specialist", 0.90) - ov.get("p_fix_security_coder", 0.15), 2
            ),
            "runs": len(rs),
            "env": {},
        }
        for env in ("nominal", "perturbed"):
            cell = {
                a: {
                    "completion": 100 * sum(1 for s in seeds if by[(a, env, s)]["success"]) / n,
                    "violations": sum(by[(a, env, s)]["violations"] for s in seeds),
                    "adaptations": round(sum(by[(a, env, s)]["adaptations"] for s in seeds) / n, 2),
                    "invocations": round(
                        sum(by[(a, env, s)]["agent_activations"] for s in seeds) / n, 1
                    ),
                }
                for a in ARMS
            }
            a2 = [bool(by[("A2", env, s)]["success"]) for s in seeds]
            b2 = [bool(by[("B2", env, s)]["success"]) for s in seeds]
            p, n10, n01 = _mcnemar(a2, b2)
            cell["A2_vs_B2"] = {
                "cliffs_delta": round(sum(a2) / n - sum(b2) / n, 4),
                "p_mcnemar_uncorrected": p,
                "discordant": [n10, n01],
            }
            entry["env"][env] = cell
        out[key] = entry
    return out


def misspecification(n: int = 30) -> dict:
    out = {}
    for key, spec in TASKS.items():
        rows = misspecification_sweep(n, spec)
        a1 = [r["A1_compliance"] for r in rows]
        a2 = [r["A2_compliance"] for r in rows]
        d = [x - y for x, y in zip(a2, a1)]
        out[key] = {
            "configurations": len(rows),
            "A1_worst_compliance": round(100 * min(a1), 1),
            "A2_worst_compliance": round(100 * min(a2), 1),
            "A1_mean_compliance": round(100 * sum(a1) / len(a1), 1),
            "A2_mean_compliance": round(100 * sum(a2) / len(a2), 1),
            "A1_configs_admitting_violation": sum(1 for r in rows if r["A1_violations"] > 0),
            "A1_worst_completion": round(100 * min(r["A1_success"] for r in rows), 1),
            "A2_worst_completion": round(100 * min(r["A2_success"] for r in rows), 1),
            "p_signed_rank_uncorrected": 1.0 if all(v == 0 for v in d) else float(wilcoxon(d).pvalue),
        }
    return out


def main(n: int = 30, out_dir: Path = RESULTS_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"n_seeds": n, "primary_task": "rest_auth",
               "campaign": campaign(n), "misspecification": misspecification(n)}
    (out_dir / "tasks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_dir / 'tasks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
