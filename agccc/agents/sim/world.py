"""The seeded world — the single most important correctness property here.

Mitigation #1 from the plan: *one draw, shared across arms*. For a given seed,
every arm must face exactly the same luck, or the arms are not comparable and
no reported difference means anything (V3 §19.2).

The mechanism is **index-stable draws**. Each random decision is keyed by
`(role, invocation_index, defect)` and derived from a hash of the seed and that
key, not from a running generator. That matters because arms consume randomness
at different rates: B1 may invoke the Coder twice where AGCCC invokes it five
times. With a shared running generator the two arms would desynchronise after
the first divergence and every later draw would differ. Keyed derivation means
arm A's third Coder attempt faces byte-identically what arm B's third Coder
attempt faces, whatever happened in between.

The draws are also *content-independent*: the table says "would this role clear
this defect on its k-th attempt", computed over the whole defect catalogue,
never over the defects that happen to be present. Drawing over the present set
would let a difference in artefact state change the random stream, reintroducing
exactly the coupling this design removes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from ...task.catalogue import DEFAULT_TASK
from ...task.spec import TaskSpec


@dataclass(frozen=True)
class WorldParams:
    """Agent competence. The two starred values are the sensitivity axes.

    The Coder is competent at functional work and weak on security; the Security
    specialist is the reverse. That asymmetry is what makes a red security suite
    escapable only by getting the specialist into the topology — which is the
    behaviour the architecture is supposed to produce.
    """

    p_defect_functional: float = 0.40
    p_defect_security: float = 0.75
    p_fix_functional_coder: float = 0.85
    p_fix_security_coder: float = 0.15          # * sensitivity axis
    p_fix_security_specialist: float = 0.90     # * sensitivity axis
    p_fix_functional_specialist: float = 0.0
    p_planner_omits_endpoint: float = 0.10

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _unit(seed: int, *key: object) -> float:
    """A uniform draw in [0,1) keyed by seed and an arbitrary tuple."""
    blob = "|".join([str(seed), *(str(k) for k in key)]).encode()
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class World:
    """Pre-determined luck for one seed. Identical for every arm."""

    def __init__(
        self,
        seed: int,
        params: WorldParams | None = None,
        spec: TaskSpec | None = None,
    ) -> None:
        self.seed = seed
        self.spec = spec or DEFAULT_TASK
        # A task carries its own competence profile. An explicit `params`
        # always wins, so the sensitivity sweep can still override one axis.
        base = WorldParams(**(self.spec.world_overrides or {}))
        self.params = params if params is not None else base

    # --- planner --------------------------------------------------------

    def planned_endpoints(self, attempt: int) -> frozenset[str]:
        """Which endpoints the plan covers. A re-plan is a fresh attempt.

        Later attempts are strictly better: a re-plan that could silently drop
        an endpoint would let re-planning *lose* work, which is not the failure
        mode under study.
        """
        units = self.spec.all_units
        if attempt >= 1:
            return units
        omitted = {
            u
            for u in sorted(units)
            if _unit(self.seed, "plan", attempt, u) < self.params.p_planner_omits_endpoint
        }
        return frozenset(units - omitted)

    # --- coder ----------------------------------------------------------

    def initial_defects(self) -> frozenset[str]:
        p = self.params
        return frozenset(
            d
            for d in self.spec.all_defects
            if _unit(self.seed, "inject", d)
            < (p.p_defect_security if self.spec.is_security(d) else p.p_defect_functional)
        )

    def coder_clears(self, attempt: int) -> frozenset[str]:
        """Defects the Coder would clear on its `attempt`-th repair pass."""
        p = self.params
        return frozenset(
            d
            for d in self.spec.all_defects
            if _unit(self.seed, "coder_fix", attempt, d)
            < (p.p_fix_security_coder if self.spec.is_security(d) else p.p_fix_functional_coder)
        )

    def specialist_clears(self, attempt: int) -> frozenset[str]:
        p = self.params
        return frozenset(
            d
            for d in self.spec.all_defects
            if _unit(self.seed, "sec_fix", attempt, d)
            < (
                p.p_fix_security_specialist
                if self.spec.is_security(d)
                else p.p_fix_functional_specialist
            )
        )

    # --- Mode C self-report (recorded, never consumed) ------------------

    def self_report(self, role: str, attempt: int) -> float:
        """A deliberately over-confident model self-report (V3 §8.3).

        Agents in the wild report high confidence in work that fails its tests.
        This exists so the calibration divergence can be reported; nothing in
        the control path may read it.
        """
        return 0.70 + 0.25 * _unit(self.seed, "selfreport", role, attempt)

    def with_params(self, **overrides) -> "World":
        return World(self.seed, replace(self.params, **overrides), self.spec)
