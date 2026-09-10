"""Statistical reporting (V3 §19).

Mann-Whitney U (two-sided, no normality assumption) with Cliff's delta as the
effect size, corrected across the hypothesis family by Holm-Bonferroni at
α = 0.05.

This replaces the one-sample *t*-test against a fixed baseline value with
Cohen's *d* that the paper currently drafts. That test assumes the baseline is a
constant; here the baseline is itself run 30 times and has its own distribution,
so a two-sample rank test is the correct instrument.

Cliff's delta is computed exactly rather than converted from U, and its
magnitude is labelled on the conventional thresholds (0.147 / 0.33 / 0.474).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import mannwhitneyu, wilcoxon


@dataclass(frozen=True)
class Comparison:
    label: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    u: float
    p_raw: float
    p_holm: float | None
    delta: float
    magnitude: str
    significant: bool | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def cliffs_delta(a: list[float], b: list[float]) -> tuple[float, str]:
    """(#(a>b) - #(a<b)) / (n_a · n_b), computed exactly."""
    if not a or not b:
        return float("nan"), "undefined"
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    d = (gt - lt) / (len(a) * len(b))
    m = abs(d)
    label = (
        "negligible" if m < 0.147 else "small" if m < 0.33 else "medium" if m < 0.474 else "large"
    )
    return d, label


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def compare(label: str, a: list[float], b: list[float]) -> Comparison:
    """`a` is the proposed arm, `b` the comparator."""
    delta, magnitude = cliffs_delta(a, b)
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        # Two identical constant samples: U is undefined and scipy warns.
        # p = 1.0 is the honest answer — there is no difference to detect.
        u, p = float("nan"), 1.0
    else:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
    return Comparison(
        label=label,
        n_a=len(a),
        n_b=len(b),
        median_a=_median(a),
        median_b=_median(b),
        u=float(u),
        p_raw=float(p),
        p_holm=None,
        delta=delta,
        magnitude=magnitude,
    )


def compare_paired(label: str, a: list[float], b: list[float]) -> Comparison:
    """Wilcoxon signed-rank for matched observations.

    Used for the misspecification sweep, where each scoring configuration
    yields one A1 and one A2 measurement under identical seeds. Those are
    *paired*: treating them as independent samples would discard the pairing
    and understate the evidence. Configurations where the two arms agree
    contribute zero difference and are dropped by the test, which is the
    correct handling — they carry no information about which arm is better.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length samples")
    delta, magnitude = cliffs_delta(a, b)
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        w, p = float("nan"), 1.0
    else:
        w, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return Comparison(
        label=label,
        n_a=len(a),
        n_b=len(b),
        median_a=_median(a),
        median_b=_median(b),
        u=float(w),
        p_raw=float(p),
        p_holm=None,
        delta=delta,
        magnitude=magnitude,
    )


def holm(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Holm-Bonferroni step-down across the hypothesis family.

    Adjusted p-values are made monotone non-decreasing in rank order, so a
    later comparison can never be reported as more significant than an earlier
    one it did not beat.
    """
    m = len(comparisons)
    order = sorted(range(m), key=lambda i: comparisons[i].p_raw)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * comparisons[i].p_raw)
        running = max(running, val)
        adjusted[i] = running
    return [
        Comparison(
            **{
                **c.to_dict(),
                "p_holm": adjusted[i],
                "significant": adjusted[i] < alpha,
            }
        )
        for i, c in enumerate(comparisons)
    ]
