"""Failure injection (V3 §19.2, Raw_planning §11).

Controlled perturbation, not waiting for bad luck. The schedule is derived from
the run seed and from nothing else, so arm *k* seed *s* faces exactly the
perturbation arm *j* seed *s* faces. Without that guarantee the arms are not
comparable and every reported difference is confounded.

Note in particular that the schedule is **not** a function of execution
progress. Deriving "inject at the third Coder invocation" would perturb a
five-invocation arm differently from a two-invocation arm; deriving it from the
step index perturbs them identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.base import CODER, SECURITY, TESTER
from ..agents.sim.world import _unit
from ..task.artefact import Artefact
from ..task.defects import SECURITY_DEFECTS, Defect


class NullInjector:
    """The nominal environment: nothing is injected."""

    enabled = False

    def unavailable_at(self, step: int) -> frozenset[str]:
        return frozenset()

    def perturb(self, artefact: Artefact, step: int) -> Artefact:
        return artefact

    def to_dict(self) -> dict:
        return {"enabled": False}


@dataclass
class Injector:
    """The perturbed environment.

    Two injections per run, both on a seed-derived schedule:

      F6  agent unavailability — one agent is down for a fixed window
      F2  a forced security regression — a security defect reappears after the
          artefact has already been repaired, which is the "late validation
          failure" case (F5) and the one that punishes an arm that cannot
          re-verify
    """

    seed: int
    enabled: bool = True

    # --- F6: unavailability ---------------------------------------------

    @property
    def down_agent(self) -> str:
        pool = (CODER, SECURITY, TESTER)
        return pool[int(_unit(self.seed, "down_who") * len(pool))]

    @property
    def down_from(self) -> int:
        return 3 + int(_unit(self.seed, "down_from") * 5)

    @property
    def down_until(self) -> int:
        return self.down_from + 2

    def unavailable_at(self, step: int) -> frozenset[str]:
        if self.down_from <= step < self.down_until:
            return frozenset({self.down_agent})
        return frozenset()

    # --- F2/F5: forced late security regression -------------------------

    @property
    def regression_at(self) -> int:
        return 5 + int(_unit(self.seed, "regress_at") * 6)

    @property
    def regression_defect(self) -> Defect:
        return SECURITY_DEFECTS[int(_unit(self.seed, "regress_what") * len(SECURITY_DEFECTS))]

    def perturb(self, artefact: Artefact, step: int) -> Artefact:
        if step == self.regression_at:
            return artefact.with_defects({self.regression_defect})
        return artefact

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "down_agent": self.down_agent,
            "down_window": [self.down_from, self.down_until],
            "regression_at": self.regression_at,
            "regression_defect": self.regression_defect.value,
        }


def for_env(env: str, seed: int):
    if env == "nominal":
        return NullInjector()
    if env == "perturbed":
        return Injector(seed)
    raise ValueError(f"unknown environment {env!r}")
