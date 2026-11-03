"""Threshold selection, honestly.

The default decision threshold of 0.50 is a convention, not a result. A model
that scores every failure 0.40 and every pass 0.10 separates the classes
perfectly -- and catches nothing at all at 0.50. The threshold is a free
parameter and it belongs to you, not to the API.

The discipline this module enforces:

  1. Choose the threshold on training data, measure it on data the threshold
     has never seen. A threshold picked on the same traces you report is
     leakage, and it is the most common way eval numbers come out too good.
  2. Set minimum precision and recall BEFORE optimising anything.
  3. If no threshold meets the minimums, SAY SO. Do not quietly relax them.
     `tune_cv` reports unqualified folds rather than hiding them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .metrics import Confusion, MetricRow, confusion_at, f_beta, precision, recall


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """Every threshold that produces a distinct classification, and no more.

    Only the midpoints between adjacent distinct scores can change the output,
    plus one threshold above the maximum (flag nothing). Sweeping a fixed grid
    of 0.01 steps both wastes work and can miss the optimum when scores cluster.
    """
    uniq = sorted(set(scores))
    if not uniq:
        return [0.5]
    cands = [uniq[0]]  # flag everything at or above the lowest score
    cands += [(a + b) / 2.0 for a, b in zip(uniq, uniq[1:])]
    cands.append(uniq[-1] + 1e-9)  # flag nothing
    return cands


def sweep(scores: Sequence[float], labels: Sequence[int]) -> list[MetricRow]:
    """Full precision/recall curve over all meaningful thresholds."""
    return [
        MetricRow.build(t, confusion_at(scores, labels, t))
        for t in candidate_thresholds(scores)
    ]


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    objective: float
    row: MetricRow

    @property
    def precision(self) -> float:
        return self.row.precision

    @property
    def recall(self) -> float:
        return self.row.recall


def best_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    beta: float = 1.0,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
) -> ThresholdChoice | None:
    """Maximise F-beta subject to floors on precision and recall.

    Returns None when no threshold satisfies both floors. That None is the
    point: it is a finding about the model, not an inconvenience. A caller that
    responds by lowering the floors has decided the model's performance by
    choosing what to require of it.

    Ties are broken toward the HIGHER threshold, which flags fewer traces. When
    two thresholds score identically, prefer the one that raises fewer alarms;
    it is cheaper to operate and degrades more gracefully as scores drift.
    """
    best: ThresholdChoice | None = None
    for t in candidate_thresholds(scores):
        c = confusion_at(scores, labels, t)
        p, r = precision(c), recall(c)
        if p < min_precision or r < min_recall:
            continue
        obj = f_beta(c, beta)
        if best is None or obj > best.objective or (
            obj == best.objective and t > best.threshold
        ):
            best = ThresholdChoice(t, obj, MetricRow.build(t, c))
    return best


def stratified_folds(
    labels: Sequence[int], k: int = 5, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """k-fold splits that preserve the failure rate in every fold.

    Plain k-fold on an eval set with 18 failures in 100 traces can hand you a
    fold with two failures, where recall moves in steps of 0.5. Stratify.

    Returns [(train_idx, test_idx), ...].
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    if len(pos) < k or len(neg) < k:
        raise ValueError(
            f"need >= {k} of each class to make {k} stratified folds; "
            f"have {len(pos)} failures and {len(neg)} passes"
        )
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    buckets: list[list[int]] = [[] for _ in range(k)]
    for group in (pos, neg):
        for j, idx in enumerate(group):
            buckets[j % k].append(idx)

    folds = []
    for i in range(k):
        test = sorted(buckets[i])
        train = sorted(idx for j, b in enumerate(buckets) if j != i for idx in b)
        folds.append((train, test))
    return folds


@dataclass
class FoldResult:
    fold: int
    threshold: float | None
    qualified_on_train: bool
    test_row: MetricRow | None
    meets_floors_on_test: bool


@dataclass
class CVReport:
    """What an honest threshold-tuning run produces."""

    model: str
    beta: float
    min_precision: float
    min_recall: float
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def n_unqualified_folds(self) -> int:
        return sum(1 for f in self.folds if not f.qualified_on_train)

    @property
    def n_failed_on_test(self) -> int:
        """Folds that qualified on train but missed the floors on held-out data.

        This count is the honest measure of how much your tuning overfitted.
        """
        return sum(
            1 for f in self.folds if f.qualified_on_train and not f.meets_floors_on_test
        )

    def _mean(self, attr: str) -> float | None:
        vals = [
            getattr(f.test_row, attr) for f in self.folds if f.test_row is not None
        ]
        return sum(vals) / len(vals) if vals else None

    @property
    def mean_test_precision(self) -> float | None:
        return self._mean("precision")

    @property
    def mean_test_recall(self) -> float | None:
        return self._mean("recall")

    @property
    def mean_test_f1(self) -> float | None:
        return self._mean("f1")

    @property
    def recommended_threshold(self) -> float | None:
        """Median of the per-fold thresholds, or None if no fold qualified.

        The median rather than the mean: threshold distributions are often
        skewed by one fold, and you want the value that is typical, not the one
        dragged by an outlier. Treat this as a starting point to confirm on a
        held-out set, never as a final answer.
        """
        ts = sorted(f.threshold for f in self.folds if f.threshold is not None)
        if not ts:
            return None
        mid = len(ts) // 2
        return ts[mid] if len(ts) % 2 else (ts[mid - 1] + ts[mid]) / 2.0

    def summary(self) -> str:
        lines = [
            f"model              {self.model}",
            f"objective          F-beta (beta={self.beta})",
            f"floors             precision >= {self.min_precision:.2f}, "
            f"recall >= {self.min_recall:.2f}",
            f"folds              {len(self.folds)}",
            f"unqualified (train){self.n_unqualified_folds:>4}",
            f"missed floors (test){self.n_failed_on_test:>3}",
        ]
        if self.recommended_threshold is None:
            lines.append("recommended        NONE -- no threshold met the floors")
        else:
            lines.append(f"recommended        {self.recommended_threshold:.4f}")
            lines.append(
                f"held-out mean      P={self.mean_test_precision:.3f} "
                f"R={self.mean_test_recall:.3f} F1={self.mean_test_f1:.3f}"
            )
        return "\n".join(lines)


def tune_cv(
    scores: Sequence[float],
    labels: Sequence[int],
    beta: float = 1.0,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    k: int = 5,
    seed: int = 0,
    model: str = "model",
    folds: Sequence[tuple[list[int], list[int]]] | None = None,
) -> CVReport:
    """Tune the threshold with stratified k-fold cross-validation.

    Pass the SAME `folds` to every model you are comparing. Different splits
    make differences between models unmeasurable, because you cannot tell a
    better model from an easier fold.
    """
    splits = list(folds) if folds is not None else stratified_folds(labels, k, seed)
    report = CVReport(
        model=model, beta=beta, min_precision=min_precision, min_recall=min_recall
    )

    for i, (train_idx, test_idx) in enumerate(splits):
        tr_s = [scores[j] for j in train_idx]
        tr_y = [labels[j] for j in train_idx]
        choice = best_threshold(tr_s, tr_y, beta, min_precision, min_recall)

        if choice is None:
            report.folds.append(FoldResult(i, None, False, None, False))
            continue

        te_s = [scores[j] for j in test_idx]
        te_y = [labels[j] for j in test_idx]
        row = MetricRow.build(choice.threshold, confusion_at(te_s, te_y, choice.threshold))
        report.folds.append(
            FoldResult(
                fold=i,
                threshold=choice.threshold,
                qualified_on_train=True,
                test_row=row,
                meets_floors_on_test=(
                    row.precision >= min_precision and row.recall >= min_recall
                ),
            )
        )
    return report


def feasible_region(
    scores: Sequence[float],
    labels: Sequence[int],
    min_precision: float,
    min_recall: float,
) -> list[MetricRow]:
    """Every threshold meeting both floors. Empty means the model cannot do the
    job, whatever threshold you pick -- which is a decision you can make before
    spending anything on tuning."""
    return [
        r
        for r in sweep(scores, labels)
        if r.precision >= min_precision and r.recall >= min_recall
    ]


def threshold_sensitivity(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    objective: Callable[[Confusion], float] | None = None,
    delta: float = 0.05,
) -> float:
    """How much the objective moves if the threshold is off by +/- delta.

    A threshold on a knife edge will not survive contact with next month's
    traces. Large sensitivity is a reason to prefer a flatter nearby optimum
    even at slightly lower measured performance.
    """
    obj = objective or (lambda c: f_beta(c, 1.0))
    here = obj(confusion_at(scores, labels, threshold))
    lo = obj(confusion_at(scores, labels, max(0.0, threshold - delta)))
    hi = obj(confusion_at(scores, labels, min(1.0 + 1e-9, threshold + delta)))
    return max(abs(here - lo), abs(here - hi))
