"""The evidence model — why the decision changes without a rule telling it to.

Raw_planning §12 Guideline 6 forbids `if retry_count == 3: invoke_security()`.
Nothing of that kind exists anywhere in this codebase. Instead each action
carries a Beta(1,1) prior updated by observed outcomes, and its Laplace-smoothed
posterior mean

    P_success(a) = (successes + 1) / (attempts + 2)

decays as the action keeps failing. A Coder that has failed three times scores
0.500 -> 0.333 -> 0.250 -> 0.200 while untried alternatives hold their 0.500
prior, so the ranking flips on its own. There is no threshold, no counter
comparison, and no escalation rule: the number simply stops being the largest.

This is the primary evidence for the paper's central claim, which is why every
candidate's score at every round is written to the audit record (V3 §20 E1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Action


@dataclass
class Beta:
    alpha: int = 1
    beta: int = 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1
        else:
            self.beta += 1


@dataclass
class EvidenceModel:
    """Per-action outcome history for one run."""

    posteriors: dict[Action, Beta] = field(default_factory=dict)

    def _get(self, action: Action) -> Beta:
        return self.posteriors.setdefault(action, Beta())

    def p_success(self, action: Action) -> float:
        return self._get(action).mean

    def record(self, action: Action, success: bool) -> None:
        self._get(action).update(success)

    def attempts(self, action: Action) -> int:
        b = self._get(action)
        return b.alpha + b.beta - 2

    def snapshot(self) -> dict[str, float]:
        return {a.value: round(self.p_success(a), 6) for a in Action}
