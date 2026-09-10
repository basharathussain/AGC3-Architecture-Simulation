"""The decision engine — generate, filter, rank, select, in that order.

    Generate candidates
        -> Policy filter          <-- inadmissible actions are DELETED here
            -> Utility / cost / risk
                -> Rank
                    -> Select

The filter runs *before* the ranker and the ranker is never handed a rejected
candidate. That is the filter-then-rank invariant (V3 §1), and the whole of
Article #2 turns on it: under filter-then-rank a prohibited action is
structurally unreachable, whereas under rank-then-check it is merely
outweighed — and anything merely outweighed can be outweighed back by a large
enough utility.

The enforcement is structural rather than documentary. `_rank` receives only the
admissible set and has no access to the rejected one, and `ScoringSpy` in the
tests asserts the scorer is never invoked on a rejected action.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..awareness.state import SituationState
from .actions import ALL_ACTIONS, Action
from .evidence import EvidenceModel
from .policies import EMPTY, PolicySet
from .scoring import DEFAULT, ScoringConfig, objective


@dataclass
class Decision:
    selected: Action | None
    scores: dict[Action, float] = field(default_factory=dict)
    rejected: dict[Action, str] = field(default_factory=dict)
    considered: tuple[Action, ...] = ()
    p_success: dict[Action, float] = field(default_factory=dict)

    @property
    def escalated(self) -> bool:
        """No admissible candidate survived the filter.

        The paper's configured safety condition: terminate or escalate. Never
        relax a policy to keep going.
        """
        return self.selected is None


class DecisionEngine:
    def __init__(
        self,
        policies: PolicySet | None = None,
        scoring: ScoringConfig = DEFAULT,
        scorer=objective,
        action_space: tuple[Action, ...] = ALL_ACTIONS,
    ) -> None:
        # Narrowing the action space is how the Table IV ablations remove
        # re-orchestration while leaving situation awareness and governance
        # intact: the arm can still observe and still refuse, it simply has no
        # structural repair available.
        self.action_space = action_space
        # `policies=EMPTY` is the A1 ungoverned arm: the filter stage still runs
        # but has nothing to remove. Scoring is byte-identical between A1 and
        # A2, so any difference between those arms is attributable to the filter
        # alone and to nothing else.
        self.policies = policies if policies is not None else EMPTY
        self.scoring = scoring
        self._scorer = scorer

    # --- stage 1: generate ---------------------------------------------

    def generate(self, state: SituationState) -> tuple[Action, ...]:
        return self.action_space

    # --- stage 2: filter -----------------------------------------------

    def filter(
        self, candidates: tuple[Action, ...], state: SituationState
    ) -> tuple[tuple[Action, ...], dict[Action, str]]:
        prohibited = self.policies.prohibited(state)
        admissible = tuple(a for a in candidates if a not in prohibited)
        rejected = {a: prohibited[a] for a in candidates if a in prohibited}
        return admissible, rejected

    # --- stage 3+4: rank and select -------------------------------------

    def _rank(
        self,
        admissible: tuple[Action, ...],
        state: SituationState,
        evidence: EvidenceModel,
        endpoints_implemented: int,
    ) -> dict[Action, float]:
        return {
            a: self._scorer(
                a, state, evidence.p_success(a), endpoints_implemented, self.scoring
            )
            for a in admissible
        }

    def decide(
        self,
        state: SituationState,
        evidence: EvidenceModel,
        endpoints_implemented: int,
    ) -> Decision:
        candidates = self.generate(state)
        admissible, rejected = self.filter(candidates, state)

        if not admissible:
            return Decision(selected=None, rejected=rejected, considered=candidates)

        scores = self._rank(admissible, state, evidence, endpoints_implemented)
        # Ties broken by the declaration order of `Action`, so a run is
        # reproducible rather than dependent on dict iteration.
        best = max(admissible, key=lambda a: (scores[a], -ALL_ACTIONS.index(a)))
        return Decision(
            selected=best,
            scores=scores,
            rejected=rejected,
            considered=candidates,
            p_success={a: evidence.p_success(a) for a in admissible},
        )
