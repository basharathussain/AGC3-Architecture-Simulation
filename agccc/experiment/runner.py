"""Experiment runner (V3 §19, §20).

Executes 5 arms × 2 environments × n seeds, holding the task, agents, world,
suites and injection schedule fixed and varying only the coordination
mechanism.

The scoring-configuration hash is registered once per experiment and checked per
run. V3 §19.1 is explicit that a run whose hash differs is *invalid and must be
discarded, not reinterpreted*, so a mismatch raises rather than warns.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.sim.roles import build_roster
from ..agents.sim.world import World, WorldParams
from ..governance.scoring import DEFAULT, ScoringConfig
from ..kernel.runtime import DEFAULT_STEP_BUDGET, Runtime
from ..task.suites import SUITE_VERSION
from . import arms as arm_registry
from . import metrics as M
from .injection import for_env

ENVIRONMENTS = ("nominal", "perturbed")
DEFAULT_N = 30

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Experiment:
    n: int = DEFAULT_N
    scoring: ScoringConfig = DEFAULT
    world_params: WorldParams = field(default_factory=WorldParams)
    step_budget: int = DEFAULT_STEP_BUDGET
    escalate_after: int = 2
    agent_backend: str = "sim"

    @property
    def scoring_hash(self) -> str:
        return self.scoring.hash()

    def run_one(self, arm: str, env: str, seed: int):
        world = World(seed, self.world_params)
        strategy = arm_registry.build(
            arm, scoring=self.scoring, escalate_after=self.escalate_after
        )
        runtime = Runtime(
            roster=build_roster(world),
            strategy=strategy,
            injector=for_env(env, seed),
            step_budget=self.step_budget,
        )
        record = runtime.run(arm, env, seed)
        return record, strategy

    def run_all(self, verbose: bool = False):
        registered = self.scoring_hash
        rows: list[M.Metrics] = []
        records: list[dict] = []
        started = time.perf_counter()

        for arm in arm_registry.ARM_ORDER:
            for env in ENVIRONMENTS:
                for seed in range(1, self.n + 1):
                    if self.scoring.hash() != registered:
                        raise RuntimeError(
                            "scoring configuration changed mid-experiment; "
                            "these runs are invalid and must be discarded"
                        )
                    record, strategy = self.run_one(arm, env, seed)
                    rows.append(M.compute(record, strategy))
                    records.append(
                        {
                            **record.to_dict(),
                            "decisions_trace": getattr(strategy, "decisions", []),
                            "decision_violations": getattr(strategy, "violations", []),
                        }
                    )
                if verbose:
                    subset = [r for r in rows if r.arm == arm and r.env == env]
                    print(
                        f"  {arm} {env:9s} success={M.rate(subset,'success'):6.1%} "
                        f"recovery={M.recovery_rate(subset):6.1%} "
                        f"compliance={M.compliance_rate(subset):6.1%}"
                    )

        return rows, records, {
            "scoring_hash": registered,
            "scoring": self.scoring.__dict__,
            "world_params": self.world_params.to_dict(),
            "n_per_arm_per_env": self.n,
            "seeds": [1, self.n],
            "arms": list(arm_registry.ARM_ORDER),
            "environments": list(ENVIRONMENTS),
            "step_budget": self.step_budget,
            "supervisor_escalate_after": self.escalate_after,
            "agent_backend": self.agent_backend,
            "suite_version": SUITE_VERSION,
            "confidence_mode": "derived (Mode B)",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_clock_s": round(time.perf_counter() - started, 3),
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agccc.experiment.runner")
    p.add_argument("--arm", choices=arm_registry.ARM_ORDER)
    p.add_argument("--env", choices=ENVIRONMENTS)
    p.add_argument("--seed", type=int)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out", default=str(RESULTS_DIR))
    args = p.parse_args(argv)

    exp = Experiment(n=args.n)

    if args.arm and args.seed is not None:
        record, strategy = exp.run_one(args.arm, args.env or "nominal", args.seed)
        m = M.compute(record, strategy)
        print(json.dumps(m.to_dict(), indent=2))
        return 0

    if not args.all:
        p.error("pass --all, or --arm with --seed")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"AGCCC experiment: {len(arm_registry.ARM_ORDER)} arms x "
          f"{len(ENVIRONMENTS)} envs x n={exp.n} = "
          f"{len(arm_registry.ARM_ORDER) * len(ENVIRONMENTS) * exp.n} runs")
    rows, records, repro = exp.run_all(verbose=True)

    (out / "metrics.json").write_text(json.dumps([r.to_dict() for r in rows], indent=1))
    (out / "runs.json").write_text(json.dumps(records, indent=1))
    (out / "reproducibility.json").write_text(json.dumps(repro, indent=2))
    print(f"\n{len(rows)} runs in {repro['wall_clock_s']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
