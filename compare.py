"""Comparing judges: cost, uncertainty, and whether a difference is real.

Three questions, in the order they matter:

  1. Is the difference between these two judges larger than noise?
       - difference in ACCURACY / error rate       -> mcnemar
       - difference in F1, precision, recall, MCC  -> paired_bootstrap_delta
  2. How wide is the interval on my headline metric?  -> bootstrap_metric
  3. Which judges are worth considering, given cost?  -> pareto_frontier

Most eval reports answer only the third and treat a 4-point F1 gap on 100
traces as decisive. On 100 traces it usually is not.

**Use the right test for the metric you are quoting.** McNemar's test is a
test of the marginal error rates of two classifiers. It is NOT a test of a
difference in F1, precision or recall: those are ratios whose denominators
themselves change between the two classifiers, and the discordant-pair count
McNemar works from does not capture that. Quoting a McNemar p-value next to an
F1 comparison is a category error -- one this package made in its first
release, and the reason `paired_bootstrap_delta` exists.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .metrics import Confusion, confusion_at, f1


# --------------------------------------------------------------------------
# Is the difference real?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A right, B wrong
    c: int  # A wrong, B right
    statistic: float
    p_value: float
    n_discordant: int

    def summary(self) -> str:
        return (
            f"A-only-right={self.b}  B-only-right={self.c}  "
            f"discordant={self.n_discordant}  p={self.p_value:.4f}"
        )


def mcnemar(
    pred_a: Sequence[int], pred_b: Sequence[int], labels: Sequence[int]
) -> McNemarResult:
    """Paired test of whether two classifiers differ in ACCURACY.

    SCOPE, because this is routinely misused: McNemar tests the null that the
    two classifiers have the same marginal error rate. It says nothing
    directly about a difference in F1, precision or recall -- for those, use
    `paired_bootstrap_delta`. Reach for McNemar when the quantity you care
    about is "how often is each judge right".

    Within that scope it is the right test, because the two judges saw
    identical traces: the comparison is paired, and an unpaired test discards
    that structure and loses power. Only the traces the judges disagree on
    carry information -- agreement, right or wrong, says nothing about which
    is better.

    Uses the exact binomial two-sided p-value, which stays valid when the
    discordant count is small. That is the common case in evals: 100 traces may
    produce eight disagreements, and the chi-square approximation is unreliable
    there.
    """
    if not (len(pred_a) == len(pred_b) == len(labels)):
        raise ValueError(
            f"length mismatch: {len(pred_a)} / {len(pred_b)} predictions, "
            f"{len(labels)} labels"
        )
    b = c = 0
    for pa, pb, y in zip(pred_a, pred_b, labels):
        a_right, b_right = (pa == y), (pb == y)
        if a_right and not b_right:
            b += 1
        elif b_right and not a_right:
            c += 1

    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, 0)

    # Exact two-sided binomial test against p = 0.5.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    p = min(1.0, 2.0 * tail)

    stat = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0  # continuity-corrected chi2
    return McNemarResult(b, c, stat, p, n)


@dataclass(frozen=True)
class DeltaResult:
    """A paired comparison of one metric between two judges."""

    delta: float  # metric(A) - metric(B) on the full sample
    lo: float
    hi: float
    metric_a: float
    metric_b: float
    n_boot: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero.

        'Significant' here means only: a paired bootstrap at this sample size
        did not include no-difference. It is not a statement about whether the
        difference matters to your product.
        """
        return (self.lo > 0.0) or (self.hi < 0.0)

    def summary(self) -> str:
        verdict = (
            "difference survives resampling"
            if self.significant
            else "NOT distinguishable at this sample size"
        )
        return (
            f"A={self.metric_a:.3f}  B={self.metric_b:.3f}  "
            f"delta={self.delta:+.3f}  95% CI [{self.lo:+.3f}, {self.hi:+.3f}]  "
            f"-> {verdict}"
        )


def paired_bootstrap_delta(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
    threshold_a: float,
    threshold_b: float,
    metric: Callable[[Confusion], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DeltaResult:
    """Interval on the DIFFERENCE in a metric between two judges.

    This is the correct tool for "is judge A's F1 really better than judge
    B's". It resamples traces once per iteration and recomputes BOTH judges on
    the same resampled traces, so the pairing is preserved and the shared
    difficulty of a given trace cancels. Two separate one-sample bootstraps,
    compared by eye for overlap, is a weaker and more conservative procedure --
    overlapping intervals do not imply no difference.

    If the interval on the difference contains zero, your data does not support
    a claim that one judge is better. Pick on cost, latency or
    maintainability, and say that is why.
    """
    if not (len(scores_a) == len(scores_b) == len(labels)):
        raise ValueError(
            f"length mismatch: {len(scores_a)} / {len(scores_b)} scores, "
            f"{len(labels)} labels"
        )
    m = metric or f1
    n = len(labels)
    ma = m(confusion_at(scores_a, labels, threshold_a))
    mb = m(confusion_at(scores_b, labels, threshold_b))
    if n == 0:
        return DeltaResult(0.0, 0.0, 0.0, ma, mb, n_boot)

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        by = [labels[i] for i in idx]
        da = m(confusion_at([scores_a[i] for i in idx], by, threshold_a))
        db = m(confusion_at([scores_b[i] for i in idx], by, threshold_b))
        deltas.append(da - db)
    deltas.sort()
    lo = deltas[max(0, int((alpha / 2) * n_boot))]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return DeltaResult(ma - mb, lo, hi, ma, mb, n_boot)


def bootstrap_metric(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    metric: Callable[[Confusion], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for any confusion-based metric.

    Returns (point_estimate, lo, hi).

    Resamples traces with replacement, recomputes the metric, and reads the
    percentiles. Works for metrics with no closed-form interval -- F1, MCC,
    balanced accuracy -- where there is no textbook formula to reach for.

    Two honest limitations to state when teaching it: the bootstrap cannot
    manufacture information that is not in your sample, so on 20 failures the
    interval will be embarrassingly wide and that width is the true message;
    and the percentile method is mildly biased for metrics near 0 or 1.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"length mismatch: {len(scores)} scores but {len(labels)} labels"
        )
    m = metric or f1
    point = m(confusion_at(scores, labels, threshold))
    n = len(scores)
    if n == 0:
        return (point, 0.0, 0.0)

    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs = [scores[i] for i in idx]
        by = [labels[i] for i in idx]
        vals.append(m(confusion_at(bs, by, threshold)))
    vals.sort()
    lo = vals[max(0, int((alpha / 2) * n_boot))]
    hi = vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (point, lo, hi)


# --------------------------------------------------------------------------
# Cost against performance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One judge under consideration. Cost is per 100 eval runs, in cents --
    the unit that makes a CI decision legible ("this suite costs $4 a day")."""

    name: str
    f1: float
    cost_cents_per_100: float
    latency_ms: float | None = None
    meets_floors: bool = True


def pareto_frontier(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Candidates not dominated on both cost and performance.

    B dominates A if B is at least as good AND no more expensive, with at least
    one strict. Everything dominated can be dropped without argument; what
    remains is a genuine trade-off for a human to decide.

    Note it says nothing about whether the frontier is worth being on. Run the
    floors first -- a cheap judge that cannot meet your minimum recall is not a
    trade-off, it is disqualified.
    """
    out = []
    for a in candidates:
        dominated = any(
            b is not a
            and b.f1 >= a.f1
            and b.cost_cents_per_100 <= a.cost_cents_per_100
            and (b.f1 > a.f1 or b.cost_cents_per_100 < a.cost_cents_per_100)
            for b in candidates
        )
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda c: c.cost_cents_per_100)


def cost_per_month(
    cost_cents_per_100: float, runs_per_day: int, days: int = 30
) -> float:
    """Cents per 100 runs is abstract. Dollars per month closes the argument."""
    return cost_cents_per_100 / 100.0 * runs_per_day * days / 100.0


def agreement(pred_a: Sequence[int], pred_b: Sequence[int]) -> float:
    """Raw agreement rate. Report cohens_kappa alongside it -- always."""
    if len(pred_a) != len(pred_b):
        raise ValueError(
            f"length mismatch: {len(pred_a)} vs {len(pred_b)} labels"
        )
    if not pred_a:
        return 1.0
    return sum(1 for x, y in zip(pred_a, pred_b) if x == y) / len(pred_a)


def cohens_kappa(pred_a: Sequence[int], pred_b: Sequence[int]) -> float:
    """Agreement above chance, in (-inf, 1].

    Essential for human labelling. Two annotators on a task where 90% of traces
    pass will agree 85% of the time by accident. Kappa removes that floor.

    Rough reading: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8
    substantial, > 0.8 near-perfect. If your two labellers score below about
    0.6, the disagreement is in your CRITERIA, not your annotators -- go and
    rewrite the rubric before labelling anything more.
    """
    if len(pred_a) != len(pred_b):
        raise ValueError(
            f"length mismatch: {len(pred_a)} vs {len(pred_b)} labels"
        )
    n = len(pred_a)
    if n == 0:
        return 1.0
    po = agreement(pred_a, pred_b)
    pa1 = sum(pred_a) / n
    pb1 = sum(pred_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1.0 - pe)
