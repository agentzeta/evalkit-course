"""Calibration: can you read the probability literally?

A score of 0.80 is calibrated if, among traces scoring about 0.80, about 80%
really do contain failures. Calibration is a different question from
discrimination: a model can rank traces perfectly and still be badly
calibrated (multiply every score by 0.5 -- the ranking is untouched, every
probability is now a lie), and a model can be perfectly calibrated and useless
(emit the base rate for every trace).

So:
  - If you only threshold the score, you need discrimination. Calibration is
    irrelevant; the threshold absorbs any monotone distortion.
  - The moment you read the number as a confidence -- routing, triage,
    expected-cost decisions, "auto-fix below 0.2, escalate above 0.9" --
    calibration becomes load-bearing.

And calibration is an empirical question. Platt scaling sometimes makes things
worse. Measure before and after; never apply it as a ritual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

EPS = 1e-6


def _clip(p: float, eps: float = EPS) -> float:
    return min(1.0 - eps, max(eps, p))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    # Branch to avoid overflow in exp for large |z|.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _paired(scores: Sequence[float], labels: Sequence[int]) -> None:
    """Reject mismatched lengths instead of silently truncating.

    `zip` stops at the shorter sequence. A metric computed over 180 of your
    200 labels because a scores list was short returns a plausible number and
    no warning, and it is wrong in a direction you cannot predict. Every
    paired function in this package validates first.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"length mismatch: {len(scores)} scores but {len(labels)} labels"
        )


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------


def brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0 is perfect.

    A proper scoring rule: it is minimised only by reporting your true belief,
    so it cannot be gamed by hedging toward 0.5 or by over-confidence. This is
    the number to report when comparing calibrations.

    Reference point: always predicting the base rate b gives b*(1-b). Beat that
    or your probabilities add nothing.
    """
    _paired(scores, labels)
    if not scores:
        return 0.0
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def log_loss(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Also proper, and far harsher on confident mistakes than Brier.

    Use log loss when a confident error is much worse than an uncertain one --
    which is usually the case for anything that triggers an automated action.
    """
    _paired(scores, labels)
    if not scores:
        return 0.0
    total = 0.0
    for s, y in zip(scores, labels):
        p = _clip(s)
        total += -(math.log(p) if y == 1 else math.log(1.0 - p))
    return total / len(scores)


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Signed miscalibration. Positive = model under-predicts failure."""
        return self.observed_rate - self.mean_pred


def reliability_bins(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 5
) -> list[Bin]:
    """Equal-COUNT bins (quantile binning), not equal-width.

    Equal-width bins put most of your traces in one or two buckets when scores
    cluster, and leave other buckets with two traces whose observed rate is
    0.0 or 1.0 -- noise that looks like miscalibration. Equal-count keeps every
    point's uncertainty comparable, which is what makes a reliability diagram
    readable.
    """
    _paired(scores, labels)
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    n = len(order)
    n_bins = max(1, min(n_bins, n))
    out: list[Bin] = []
    base = n // n_bins
    extra = n % n_bins
    start = 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        if size == 0:
            continue
        idx = order[start : start + size]
        start += size
        ss = [scores[i] for i in idx]
        ys = [labels[i] for i in idx]
        out.append(
            Bin(
                lo=min(ss),
                hi=max(ss),
                n=len(idx),
                mean_pred=sum(ss) / len(ss),
                observed_rate=sum(ys) / len(ys),
            )
        )
    return out


def ece(scores: Sequence[float], labels: Sequence[int], n_bins: int = 5) -> float:
    """Expected calibration error: average |predicted - observed|, weighted by
    bin size.

    Intuitive and widely reported, but it depends on the binning and it is not
    a proper scoring rule -- you can lower ECE by making predictions worse.
    Report it next to Brier, never instead of it.
    """
    _paired(scores, labels)
    bins = reliability_bins(scores, labels, n_bins)
    if not bins:
        return 0.0
    n = sum(b.n for b in bins)
    return sum(b.n * abs(b.gap) for b in bins) / n


def brier_decomposition(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 5
) -> tuple[float, float, float]:
    """Murphy's decomposition: Brier ~= reliability - resolution + uncertainty.

      reliability  how far each bin's predictions sit from its outcome rate.
                   LOWER is better. This is the part calibration can fix.
      resolution   how far bin outcome rates spread from the base rate.
                   HIGHER is better. This is real discriminating information,
                   and post-hoc calibration cannot create it.
      uncertainty  b*(1-b) for base rate b. A property of your data, not your
                   model. Nothing you do changes it.

    The teaching point: if resolution is near zero, your model knows nothing and
    calibration will only make its ignorance better-dressed.

    (Equality is exact only for the binned estimate; small residuals are normal.)
    """
    _paired(scores, labels)
    if not scores:
        return (0.0, 0.0, 0.0)
    n = len(scores)
    base = sum(labels) / n
    bins = reliability_bins(scores, labels, n_bins)
    reliability = sum(b.n * (b.mean_pred - b.observed_rate) ** 2 for b in bins) / n
    resolution = sum(b.n * (b.observed_rate - base) ** 2 for b in bins) / n
    uncertainty = base * (1.0 - base)
    return (reliability, resolution, uncertainty)


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


@dataclass
class PlattScaler:
    """Platt scaling: fit p_cal = sigmoid(a * logit(p) + b).

    Two parameters, so it is stable on small eval sets -- which is why it is the
    default choice when you have 100 labelled traces rather than 10,000. The
    cost is rigidity: it can only stretch and shift a sigmoid, so it cannot fix
    a non-monotone or S-shaped miscalibration, and on a model that is already
    well calibrated it can make matters slightly worse.

    Uses Platt's original smoothed targets rather than raw 0/1, which prevents
    the fit from being dragged to the extremes on small samples:
        y+ = (N+ + 1) / (N+ + 2),    y- = 1 / (N- + 2)
    """

    a: float = 1.0
    b: float = 0.0
    n_iter: int = 100
    fitted: bool = False

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> PlattScaler:
        _paired(scores, labels)
        n_pos = sum(1 for y in labels if y == 1)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            # Nothing to learn from a single-class sample. Stay the identity.
            self.a, self.b, self.fitted = 1.0, 0.0, True
            return self

        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        xs = [_logit(s) for s in scores]
        ts = [hi if y == 1 else lo for y in labels]

        a, b = 1.0, 0.0
        for _ in range(self.n_iter):
            # Gradient and Hessian of the negative log-likelihood.
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for x, t in zip(xs, ts):
                p = _sigmoid(a * x + b)
                d = p - t
                w = max(p * (1.0 - p), 1e-12)
                g_a += d * x
                g_b += d
                h_aa += w * x * x
                h_ab += w * x
                h_bb += w
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            # Newton step: solve H @ step = g, then subtract.
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (h_aa * g_b - h_ab * g_a) / det
            a -= step_a
            b -= step_b
            if abs(step_a) < 1e-10 and abs(step_b) < 1e-10:
                break

        self.a, self.b, self.fitted = a, b, True
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("call fit() before transform()")
        return [_sigmoid(self.a * _logit(s) + self.b) for s in scores]


@dataclass
class IsotonicCalibrator:
    """Isotonic regression via Pool Adjacent Violators (PAVA).

    Fits any monotone mapping, so it corrects shapes Platt cannot. The trade is
    overfitting: with 100 traces it will happily carve a step function out of
    noise, and it can never extrapolate beyond the score range it saw. Prefer
    Platt below a few hundred labels, isotonic above a few thousand, and always
    compare under cross-validation rather than choosing from theory.
    """

    xs: list[float] | None = None
    ys: list[float] | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        """Fit the monotone step function.

        TIED SCORES ARE AGGREGATED FIRST, and this is not optional. A
        calibrator is a function of the score: two traces scoring exactly 0.42
        must map to the same calibrated value. If tied inputs are fed to PAVA
        as separate one-point blocks they can land in different blocks with
        different values, and the fit then depends on the order the data
        happened to arrive in. Ties are common in practice because judges
        round, cache and saturate.
        """
        if len(scores) != len(labels):
            raise ValueError(f"{len(scores)} scores but {len(labels)} labels")
        if not scores:
            self.xs, self.ys = [], []
            return self

        # Aggregate by unique score, so each distinct x enters PAVA once.
        by_x: dict[float, list[float]] = {}
        for s, y in zip(scores, labels):
            by_x.setdefault(float(s), []).append(float(y))

        # Each block holds [sum_of_labels, count, max_score_in_block].
        blocks: list[list[float]] = []
        for s in sorted(by_x):
            ys = by_x[s]
            blocks.append([sum(ys), float(len(ys)), s])
            # Pool while the previous block's mean exceeds this one's.
            while len(blocks) >= 2:
                b1, b0 = blocks[-1], blocks[-2]
                if b0[0] / b0[1] <= b1[0] / b1[1]:
                    break
                blocks[-2] = [b0[0] + b1[0], b0[1] + b1[1], b1[2]]
                blocks.pop()

        self.xs = [b[2] for b in blocks]
        self.ys = [b[0] / b[1] for b in blocks]
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        if self.xs is None or self.ys is None:
            raise RuntimeError("call fit() before transform()")
        if not self.xs:
            return [0.0] * len(scores)
        out = []
        for s in scores:
            # Step function: the first block whose upper bound reaches s.
            # Scores above the largest block clamp to the last value --
            # isotonic cannot extrapolate beyond the range it was fitted on,
            # which is one of its real costs against Platt.
            val = self.ys[-1]
            for x, y in zip(self.xs, self.ys):
                if s <= x:
                    val = y
                    break
            out.append(val)
        return out


# --------------------------------------------------------------------------
# Honest evaluation of a calibrator
# --------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    model: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float

    @property
    def improved(self) -> bool:
        return self.brier_after < self.brier_before

    def summary(self) -> str:
        verdict = "improved" if self.improved else "NOT improved (keep raw scores)"
        return (
            f"{self.model:<22} Brier {self.brier_before:.4f} -> "
            f"{self.brier_after:.4f}   ECE {self.ece_before:.4f} -> "
            f"{self.ece_after:.4f}   {verdict}"
        )


def evaluate_calibrator(
    scores: Sequence[float],
    labels: Sequence[int],
    folds: Sequence[tuple[list[int], list[int]]],
    calibrator_factory=PlattScaler,
    model: str = "model",
    n_bins: int = 5,
) -> CalibrationReport:
    """Cross-validated calibration: fit the mapping on train, score on held-out.

    Fitting a calibrator and then measuring Brier on the same traces always
    looks like an improvement. It is the same leakage as tuning a threshold on
    your test set, and it is why published calibration results are so often
    irreproducible.
    """
    cal_scores: list[float] = []
    cal_labels: list[int] = []
    raw_scores: list[float] = []

    for train_idx, test_idx in folds:
        tr_s = [scores[i] for i in train_idx]
        tr_y = [labels[i] for i in train_idx]
        te_s = [scores[i] for i in test_idx]
        te_y = [labels[i] for i in test_idx]

        cal = calibrator_factory()
        cal.fit(tr_s, tr_y)
        cal_scores.extend(cal.transform(te_s))
        cal_labels.extend(te_y)
        raw_scores.extend(te_s)

    return CalibrationReport(
        model=model,
        brier_before=brier(raw_scores, cal_labels),
        brier_after=brier(cal_scores, cal_labels),
        ece_before=ece(raw_scores, cal_labels, n_bins),
        ece_after=ece(cal_scores, cal_labels, n_bins),
    )
