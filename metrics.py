"""Classifier metrics, from the confusion matrix up.

Everything an eval needs sits on a 2x2 table. This module builds that table and
derives from it, with no dependencies, so you can read every line.

Convention throughout this package:
    label 1 = FAILURE (the thing we are hunting)
    label 0 = PASS
    a trace is FLAGGED when score >= threshold

That convention matters. "Positive" means "failure" here, so recall is the share
of real failures we caught, and precision is the share of our alarms that were
real. Get this backwards and every number in your eval report inverts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Confusion:
    """A 2x2 table. tp/fn concern real failures; fp/tn concern real passes."""

    tp: int  # real failure, flagged          -> caught it
    fp: int  # real pass, flagged             -> false alarm
    fn: int  # real failure, not flagged      -> missed it
    tn: int  # real pass, not flagged         -> correctly quiet

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def n_positive(self) -> int:
        """How many real failures are in the data."""
        return self.tp + self.fn

    @property
    def n_flagged(self) -> int:
        return self.tp + self.fp

    def __str__(self) -> str:
        return (
            f"             flagged  quiet\n"
            f"  failure  {self.tp:8d} {self.fn:6d}\n"
            f"  pass     {self.fp:8d} {self.tn:6d}"
        )


def confusion_at(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> Confusion:
    """Build the table by flagging every trace scoring >= threshold.

    `>=` not `>`: a model that emits exactly 0.5 on a failure should be caught by
    a 0.5 threshold. The choice is arbitrary but it must be consistent between
    tuning and reporting, or your tuned threshold will not reproduce.
    """
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores but {len(labels)} labels")
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        flagged = s >= threshold
        if y == 1:
            tp += flagged
            fn += not flagged
        elif y == 0:
            fp += flagged
            tn += not flagged
        else:
            raise ValueError(f"labels must be 0 or 1, got {y!r}")
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def precision(c: Confusion) -> float:
    """Of the traces we flagged, what share were real failures?

    Undefined when nothing was flagged. We return 0.0, which is the
    conservative reading for eval purposes: a model that flags nothing has
    earned no credit. Be aware that sklearn returns 0.0 with a warning here and
    some texts return 1.0 -- the choice changes tuned thresholds at the extremes.
    """
    return c.tp / c.n_flagged if c.n_flagged else 0.0


def recall(c: Confusion) -> float:
    """Of the real failures, what share did we catch? (= sensitivity, TPR)"""
    return c.tp / c.n_positive if c.n_positive else 0.0


def specificity(c: Confusion) -> float:
    """Of the real passes, what share did we leave alone? (= TNR)"""
    denom = c.tn + c.fp
    return c.tn / denom if denom else 0.0


def false_positive_rate(c: Confusion) -> float:
    return 1.0 - specificity(c)


def accuracy(c: Confusion) -> float:
    """Share of all traces classified correctly.

    Almost always the wrong metric for evals. If 5% of your traces fail, a model
    that flags nothing scores 95% accuracy and is worthless. Reported here only
    so you can demonstrate that trap to a room.
    """
    return (c.tp + c.tn) / c.n if c.n else 0.0


def balanced_accuracy(c: Confusion) -> float:
    """Mean of recall and specificity. Immune to the imbalance trap above."""
    return (recall(c) + specificity(c)) / 2.0


def f_beta(c: Confusion, beta: float = 1.0) -> float:
    """Weighted harmonic mean of precision and recall.

    beta = 1   -> F1, equal weight
    beta < 1   -> precision matters more (false alarms are expensive)
    beta > 1   -> recall matters more (misses are expensive)

    beta is interpretable: recall is weighted beta^2 times as heavily as
    precision. beta=2 means a miss hurts four times as much as a false alarm.
    Pick beta from your actual cost of each error, not from habit.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    p, r = precision(c), recall(c)
    if p == 0.0 and r == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def f1(c: Confusion) -> float:
    return f_beta(c, 1.0)


def mcc(c: Confusion) -> float:
    """Matthews correlation coefficient, in [-1, 1].

    The single most honest one-number summary of a 2x2 table: it uses all four
    cells, so unlike F1 it cannot be inflated by ignoring true negatives. Worth
    reporting alongside F1 when someone asks for "the number".
    """
    num = c.tp * c.tn - c.fp * c.fn
    den = math.sqrt(
        float(c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    )
    return num / den if den else 0.0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion k/n. Use this, not k/n +- 1.96*se.

    The textbook normal interval breaks exactly where evals live: small n and
    proportions near 0 or 1. It can return bounds below 0 or above 1 and its
    coverage is poor. Wilson behaves at the edges and costs one extra line.

    z = 1.96 -> 95%, z = 1.645 -> 90%, z = 2.576 -> 99%.

    Teach this early. "Recall went from 0.82 to 0.87" on n=50 failures is a
    difference of four traces and the intervals overlap heavily.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def recall_interval(c: Confusion, z: float = 1.96) -> tuple[float, float]:
    """95% interval on recall. Denominator is the number of real failures --
    usually the smallest number in your eval set, and therefore the widest
    interval in your report."""
    return wilson_interval(c.tp, c.n_positive, z)


def precision_interval(c: Confusion, z: float = 1.96) -> tuple[float, float]:
    return wilson_interval(c.tp, c.n_flagged, z)


@dataclass(frozen=True)
class MetricRow:
    """Everything you would put in one row of an eval report."""

    threshold: float
    confusion: Confusion
    precision: float
    recall: float
    f1: float
    specificity: float
    mcc: float

    @classmethod
    def build(cls, threshold: float, c: Confusion) -> MetricRow:
        return cls(
            threshold=threshold,
            confusion=c,
            precision=precision(c),
            recall=recall(c),
            f1=f1(c),
            specificity=specificity(c),
            mcc=mcc(c),
        )


def metric_row(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> MetricRow:
    return MetricRow.build(threshold, confusion_at(scores, labels, threshold))
