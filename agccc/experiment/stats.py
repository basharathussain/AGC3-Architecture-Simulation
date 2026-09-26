"""Statistical reporting (V3 §19).

Mann-Whitney U (two-sided, no normality assumption) with Cliff's delta as the
effect size, corrected across the hypothesis family by Holm-Bonferroni at
α = 0.05.

This replaces the one-sample *t*-test against a fixed baseline value with
Cohen's *d* that the paper currently drafts. That test assumes the baseline is a
constant; here the baseline is itself run 30 times and has its own distribution,
so a two-sample rank test is the correct instrument.

**Paired where the design is paired.** The runner builds one `World(seed)` and
hands it to every arm, so arm *k* at seed *s* faces byte-identically what arm *j*
at seed *s* faces (`runner.py`). Those observations are matched, and the argument
`compare_paired` already makes for the misspecification sweep -- that treating
matched observations as independent discards the pairing and understates the
evidence -- applies to the arm comparisons too. `compare_paired_binary` is the
instrument for the binary outcomes (completion, recovery); the unpaired
`compare` is retained so both can be reported side by side.

Cliff's delta is computed exactly rather than converted from U, and its
magnitude is labelled on the conventional thresholds (0.147 / 0.33 / 0.474).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import binomtest, mannwhitneyu, wilcoxon


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
    # The same comparison computed as two independent samples, retained so the
    # paper can report the test as submitted beside the corrected one. None for
    # rows that were never unpaired.
    p_unpaired: float | None = None
    # Which instrument produced this row. Without it a mixed family cannot be
    # captioned honestly -- the table used to describe every row as Mann-Whitney
    # while one of them was a signed-rank test.
    test: str = "mannwhitney"

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
        test="mannwhitney",
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
        test="wilcoxon",
    )


def compare_paired_binary(label: str, a: list[float], b: list[float]) -> Comparison:
    """McNemar's exact test for matched binary outcomes.

    Completion and recovery are 0/1, and the seeds are matched, so the evidence
    lives entirely in the DISCORDANT pairs: seeds where one arm succeeded and the
    other did not. Concordant pairs -- both succeeded, or both failed -- say
    nothing about which arm is better and are correctly ignored.

    Exact rather than the chi-square approximation: with seven discordant pairs
    the asymptotic form is not trustworthy, and an exact binomial costs nothing
    at this scale.

    A signed-rank test on 0/1 data degenerates to this same sign test, so this is
    `compare_paired` specialised rather than a different claim -- but stating it
    as McNemar makes the instrument legible to a reader checking the design.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length samples")
    n = len(a)
    b_wins = sum(1 for x, y in zip(a, b) if x and not y)
    c_wins = sum(1 for x, y in zip(a, b) if y and not x)
    discordant = b_wins + c_wins

    if discordant == 0:
        # The arms never disagree on any seed. There is no evidence either way.
        w, p = float("nan"), 1.0
    else:
        w = float(b_wins)
        p = float(binomtest(b_wins, discordant, 0.5, alternative="two-sided").pvalue)

    # Matched-pairs effect size: the net proportion of seeds on which the
    # proposed arm wins. For binary outcomes this coincides exactly with the
    # independent-samples Cliff's delta -- both reduce to the difference in
    # proportions -- so correcting the test does not move the reported effect
    # size. Computed here in its matched form so that the number reported
    # alongside a paired test is itself a paired statistic.
    delta = (b_wins - c_wins) / n if n else float("nan")
    m = abs(delta)
    magnitude = (
        "negligible" if m < 0.147 else "small" if m < 0.33 else "medium" if m < 0.474 else "large"
    )
    return Comparison(
        label=label,
        n_a=n,
        n_b=n,
        median_a=_median(a),
        median_b=_median(b),
        u=w,
        p_raw=p,
        p_holm=None,
        delta=delta,
        magnitude=magnitude,
        test="mcnemar",
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
