"""Test suite for evalkit. Run with:  python3 -I tests/test_evalkit.py

No test framework: a tiny runner, so the suite works anywhere Python does.
Known-value tests are hand-computed and the arithmetic is in the comments, so
a student can verify the library rather than trusting it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (  # noqa: E402
    Candidate,
    IsotonicCalibrator,
    PlattScaler,
    accuracy,
    agreement,
    best_threshold,
    bootstrap_metric,
    brier,
    brier_decomposition,
    candidate_thresholds,
    cohens_kappa,
    confusion_at,
    ece,
    evaluate_calibrator,
    f1,
    f_beta,
    feasible_region,
    log_loss,
    make_dataset,
    mcc,
    mcnemar,
    metric_row,
    paired_bootstrap_delta,
    pareto_frontier,
    precision,
    recall,
    reliability_bins,
    specificity,
    stratified_folds,
    subsample,
    sweep,
    tune_cv,
    wilson_interval,
)

_PASS = 0
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAIL.append(f"{name}: {detail}" if detail else name)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _try_raises(fn, exc=ValueError) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


# ==========================================================================
# metrics
# ==========================================================================

# Reference table used throughout: tp=5, fp=1, fn=2, tn=12  (n=20)
#   precision = 5/6       = 0.833333
#   recall    = 5/7       = 0.714286
#   F1        = 2*5/(2*5+1+2) = 10/13 = 0.769231
#   spec      = 12/13     = 0.923077
#   accuracy  = 17/20     = 0.85
#   MCC       = (5*12-1*2)/sqrt(6*7*13*14) = 58/sqrt(7644) = 0.663385
_S = [0.9, 0.8, 0.7, 0.6, 0.55, 0.52] + [0.4] * 2 + [0.3] * 12
_Y = [1, 1, 1, 1, 1, 0] + [1, 1] + [0] * 12

c_ref = confusion_at(_S, _Y, 0.5)

check("confusion counts", (c_ref.tp, c_ref.fp, c_ref.fn, c_ref.tn) == (5, 1, 2, 12),
      f"got {(c_ref.tp, c_ref.fp, c_ref.fn, c_ref.tn)}")
check("confusion n", c_ref.n == 20)
check("n_positive", c_ref.n_positive == 7)
check("precision", close(precision(c_ref), 5 / 6))
check("recall", close(recall(c_ref), 5 / 7))
check("specificity", close(specificity(c_ref), 12 / 13))
check("accuracy", close(accuracy(c_ref), 0.85))
check("f1 == 10/13", close(f1(c_ref), 10 / 13))
check("mcc known value", close(mcc(c_ref), 58 / math.sqrt(7644), 1e-9))

# F-beta direction: precision (0.833) exceeds recall (0.714) here, so weighting
# precision more heavily (beta<1) must raise the score and weighting recall
# more heavily (beta>1) must lower it.
check("f_beta ordering", f_beta(c_ref, 0.5) > f1(c_ref) > f_beta(c_ref, 2.0),
      f"{f_beta(c_ref, 0.5):.4f} {f1(c_ref):.4f} {f_beta(c_ref, 2.0):.4f}")
check("f2 known value", close(f_beta(c_ref, 2.0), 0.735294, 1e-5))
check("f0.5 known value", close(f_beta(c_ref, 0.5), 0.806452, 1e-5))

# Degenerate cases must not raise.
c_none = confusion_at([0.1, 0.2], [0, 1], 0.9)  # flags nothing
check("precision of empty flag set", precision(c_none) == 0.0)
check("f1 of empty flag set", f1(c_none) == 0.0)
check("mcc degenerate is 0", mcc(confusion_at([0.1, 0.2], [0, 0], 0.05)) == 0.0)

# The accuracy trap: 95 passes, 5 failures, flag nothing -> 95% accuracy, 0 recall.
trap_s = [0.1] * 100
trap_y = [1] * 5 + [0] * 95
c_trap = confusion_at(trap_s, trap_y, 0.5)
check("accuracy trap", close(accuracy(c_trap), 0.95) and recall(c_trap) == 0.0)

# Wilson interval, 8/10 at 95% -> (0.4901, 0.9433)
lo, hi = wilson_interval(8, 10)
check("wilson lower", close(lo, 0.490, 1e-3), f"{lo:.4f}")
check("wilson upper", close(hi, 0.943, 1e-3), f"{hi:.4f}")
check("wilson bounded 0/5", wilson_interval(0, 5)[0] == 0.0)
check("wilson bounded 5/5", wilson_interval(5, 5)[1] == 1.0)
check("wilson n=0 is uninformative", wilson_interval(0, 0) == (0.0, 1.0))
# Wilson must stay inside [0,1] where the naive normal interval does not.
nlo, nhi = wilson_interval(1, 10)
check("wilson no negative bound", nlo >= 0.0)

check("metric_row builds", close(metric_row(_S, _Y, 0.5).f1, 10 / 13))


# ==========================================================================
# thresholds
# ==========================================================================

cands = candidate_thresholds([0.2, 0.5, 0.5, 0.8])
# distinct scores {0.2, 0.5, 0.8} -> [0.2, 0.35, 0.65, 0.8+eps]
check("candidate count", len(cands) == 4, f"{cands}")
check("candidate lowest flags all", cands[0] == 0.2)
check("candidate highest flags none",
      confusion_at([0.2, 0.5, 0.8], [1, 1, 1], cands[-1]).tp == 0)

rows = sweep(_S, _Y)
check("sweep covers thresholds", len(rows) == len(candidate_thresholds(_S)))
check("sweep recall is monotone non-increasing in threshold",
      all(a.recall >= b.recall for a, b in zip(rows, rows[1:])))

# best_threshold must refuse rather than relax.
imp = best_threshold([0.1, 0.9], [0, 1], min_precision=0.99, min_recall=0.99)
check("floors reachable returns a choice", imp is not None)
impossible = best_threshold([0.5, 0.5], [0, 1], min_precision=0.9, min_recall=0.9)
check("unreachable floors return None", impossible is None)
check("feasible_region empty when unreachable",
      feasible_region([0.5, 0.5], [0, 1], 0.9, 0.9) == [])

# Tie-break toward the higher threshold (fewer alarms).
tie = best_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
check("perfect separation found", tie is not None and close(tie.objective, 1.0))

# stratified folds
labels5 = [1] * 20 + [0] * 60
folds = stratified_folds(labels5, k=5, seed=1)
check("fold count", len(folds) == 5)
all_test = sorted(i for _, te in folds for i in te)
check("folds partition the data", all_test == list(range(80)))
check("no train/test overlap",
      all(not (set(tr) & set(te)) for tr, te in folds))
check("every fold keeps the base rate",
      all(close(sum(labels5[i] for i in te) / len(te), 0.25, 0.08) for _, te in folds),
      str([sum(labels5[i] for i in te) for _, te in folds]))
try:
    stratified_folds([1, 0, 0, 0], k=5)
    check("too-few-of-class raises", False)
except ValueError:
    check("too-few-of-class raises", True)


# ==========================================================================
# the headline teaching case: judge_shifted
# ==========================================================================

ds = make_dataset()
check("dataset size", len(ds) == 200)
check("base rate near 0.25", 0.18 <= ds.base_rate <= 0.32, f"{ds.base_rate:.3f}")
check("judge list", len(ds.judges) == 5)

shifted = ds.scores("judge_shifted")
check("shifted scores all below 0.5", max(shifted) < 0.5, f"max={max(shifted):.3f}")

c_default = confusion_at(shifted, ds.labels, 0.50)
check("at 0.50 judge_shifted catches nothing", c_default.tp == 0)
check("at 0.50 judge_shifted recall is zero", recall(c_default) == 0.0)

tuned = best_threshold(shifted, ds.labels)
check("tuned judge_shifted is strong", tuned is not None and tuned.objective > 0.75,
      f"F1={tuned.objective:.3f}" if tuned else "None")

# Same column, two thresholds: 0.0 against something good. That delta is the
# entire argument for tuning, and it must hold for the course to be honest.
check("tuning moves F1 from 0 to usable",
      f1(c_default) == 0.0 and tuned is not None and tuned.objective > 0.75)


# ==========================================================================
# cross-validation and leakage
# ==========================================================================

strong = ds.scores("judge_strong")
rep = tune_cv(strong, ds.labels, min_precision=0.70, min_recall=0.70,
              k=5, seed=7, model="judge_strong")
check("cv has 5 folds", len(rep.folds) == 5)
check("cv recommends a threshold", rep.recommended_threshold is not None)
check("cv summary renders", "judge_strong" in rep.summary())
check("cv mean test f1 is sane", 0.5 < (rep.mean_test_f1 or 0) <= 1.0,
      f"{rep.mean_test_f1}")

# Leakage: a threshold tuned on ALL the data and scored on ALL the data is
# optimistic relative to held-out performance. This is the single most common
# way eval numbers come out too good, so it is asserted rather than asserted-at.
in_sample = best_threshold(strong, ds.labels)
check("in-sample F1 >= held-out mean F1",
      in_sample is not None and in_sample.objective >= (rep.mean_test_f1 or 0) - 1e-9,
      f"in={in_sample.objective:.4f} cv={rep.mean_test_f1:.4f}" if in_sample else "")

# A model that cannot meet the floors must produce an honest, empty report.
weak = ds.scores("judge_weak")
hard = tune_cv(weak, ds.labels, min_precision=0.95, min_recall=0.95,
               k=5, seed=7, model="judge_weak")
check("impossible floors leave folds unqualified", hard.n_unqualified_folds == 5,
      f"{hard.n_unqualified_folds}")
check("impossible floors recommend nothing", hard.recommended_threshold is None)
check("impossible floors say so in the summary", "NONE" in hard.summary())

# Shared folds: comparing models on different splits is not a comparison.
shared = stratified_folds(ds.labels, k=5, seed=99)
ra = tune_cv(strong, ds.labels, folds=shared, model="a")
rb = tune_cv(ds.scores("judge_oracle"), ds.labels, folds=shared, model="b")
check("shared folds give equal fold counts", len(ra.folds) == len(rb.folds) == 5)


# ==========================================================================
# calibration
# ==========================================================================

check("brier known value", close(brier([0.9, 0.1], [1, 0]), 0.01))
check("brier perfect is 0", brier([1.0, 0.0], [1, 0]) == 0.0)
check("brier worst is 1", close(brier([0.0, 1.0], [1, 0]), 1.0))
# Base-rate baseline: predicting b everywhere scores b(1-b).
check("base-rate brier", close(brier([0.25] * 4, [1, 0, 0, 0]), 0.25 * 0.75))

bins = reliability_bins(strong, ds.labels, n_bins=5)
check("bin count", len(bins) == 5)
check("bins cover every trace", sum(b.n for b in bins) == 200)
check("equal-count bins are balanced", max(b.n for b in bins) - min(b.n for b in bins) <= 1)
check("bins are ordered by score",
      all(a.mean_pred <= b.mean_pred for a, b in zip(bins, bins[1:])))
check("ece in range", 0.0 <= ece(strong, ds.labels) <= 1.0)

rel, res, unc = brier_decomposition(strong, ds.labels, n_bins=5)
check("uncertainty equals base-rate variance",
      close(unc, ds.base_rate * (1 - ds.base_rate), 1e-9))
# Murphy: brier ~= reliability - resolution + uncertainty, up to binning error.
check("brier decomposition reconstructs",
      close(brier(strong, ds.labels), rel - res + unc, 0.02),
      f"brier={brier(strong, ds.labels):.4f} rel-res+unc={rel - res + unc:.4f}")
check("a discriminating judge has positive resolution", res > 0.02, f"{res:.4f}")
check("a weak judge has less resolution than a strong one",
      brier_decomposition(weak, ds.labels)[1] < res)

# The oracle reports the latent propensity the labels were drawn from, so it is
# calibrated BY CONSTRUCTION. Its reliability term must be near zero -- this
# test is what certifies the dataset can teach calibration honestly.
oracle = ds.scores("judge_oracle")
o_rel, o_res, _ = brier_decomposition(oracle, ds.labels, 5)
check("oracle reliability is near zero", o_rel < 0.01, f"{o_rel:.5f}")
check("oracle has the most resolution of any judge",
      all(o_res >= brier_decomposition(ds.scores(j), ds.labels, 5)[1] - 1e-9
          for j in ds.judges),
      f"{o_res:.4f}")
check("oracle has the lowest Brier of any judge",
      all(brier(oracle, ds.labels) <= brier(ds.scores(j), ds.labels) + 1e-9
          for j in ds.judges))
# Labels are drawn from pi, not assigned, so even the oracle cannot reach F1=1.
# That ceiling is a property of the data and no judge can beat it.
o_best = best_threshold(oracle, ds.labels)
check("oracle is strong but not perfect",
      o_best is not None and 0.70 < o_best.objective < 1.0,
      f"{o_best.objective:.4f}" if o_best else "None")
check("no judge beats the oracle",
      all((best_threshold(ds.scores(j), ds.labels) or o_best).objective
          <= o_best.objective + 1e-9 for j in ds.judges),
      str({j: round((best_threshold(ds.scores(j), ds.labels) or o_best).objective, 3)
           for j in ds.judges}))
check("propensity is a probability",
      all(0.0 <= p <= 1.0 for p in ds.propensities))
check("base rate tracks mean propensity",
      abs(sum(ds.propensities) / len(ds) - ds.base_rate) < 0.08,
      f"mean pi={sum(ds.propensities) / len(ds):.3f} base={ds.base_rate:.3f}")

# Calibrating an already-calibrated judge should NOT help. This is the honest
# case the course promises, and it must be true in the data or the lesson is a
# lie.
oracle_cal = evaluate_calibrator(oracle, ds.labels, shared, PlattScaler, "oracle")
check("platt does not improve the oracle", not oracle_cal.improved,
      oracle_cal.summary())

# Platt scaling on a deliberately miscalibrated model.
mis = [min(0.98, s * 0.45) for s in strong]  # squash: ranking kept, probs now wrong
p = PlattScaler().fit(mis, ds.labels)
fixed = p.transform(mis)
check("platt improves a squashed model",
      brier(fixed, ds.labels) < brier(mis, ds.labels),
      f"{brier(mis, ds.labels):.4f} -> {brier(fixed, ds.labels):.4f}")
check("platt output is a probability", all(0.0 <= v <= 1.0 for v in fixed))
check("platt preserves ranking (monotone)",
      all(
          (mis[i] <= mis[j]) == (fixed[i] <= fixed[j])
          for i in range(0, 60, 7)
          for j in range(0, 60, 11)
          if mis[i] != mis[j]
      ))
check("platt on single-class data stays identity",
      close(PlattScaler().fit([0.3, 0.6], [1, 1]).a, 1.0))
try:
    PlattScaler().transform([0.5])
    check("platt refuses transform before fit", False)
except RuntimeError:
    check("platt refuses transform before fit", True)

# Isotonic: output must be non-decreasing in the input.
iso = IsotonicCalibrator().fit(mis, ds.labels)
pairs = sorted(zip(mis, iso.transform(mis)))
check("isotonic output is non-decreasing",
      all(a[1] <= b[1] + 1e-12 for a, b in zip(pairs, pairs[1:])))
check("isotonic output is a probability",
      all(0.0 <= v <= 1.0 for v in iso.transform(mis)))

# Cross-validated calibration: fit on train, measure on held-out.
cal_rep = evaluate_calibrator(mis, ds.labels, shared, PlattScaler, "squashed")
check("cv calibration improves the squashed model", cal_rep.improved,
      cal_rep.summary())
check("calibration report renders", "Brier" in cal_rep.summary())
# And the honest case: calibration is not guaranteed to help.
cal_ok = evaluate_calibrator(
    ds.scores("judge_oracle"), ds.labels, shared, PlattScaler, "already-ok"
)
check("calibration verdict is reported either way",
      isinstance(cal_ok.improved, bool))


# ==========================================================================
# comparison
# ==========================================================================

# McNemar, b=10 c=0: exact two-sided p = 2 * C(10,0)/2^10 = 2/1024 = 0.001953
r = mcnemar([1] * 10, [0] * 10, [1] * 10)
check("mcnemar b/c split", (r.b, r.c) == (10, 0))
check("mcnemar exact p", close(r.p_value, 2 / 1024, 1e-9), f"{r.p_value}")
check("mcnemar no discordance is p=1", mcnemar([1, 0], [1, 0], [1, 0]).p_value == 1.0)
check("mcnemar ignores shared errors",
      mcnemar([1, 1], [1, 1], [0, 0]).n_discordant == 0)
check("mcnemar summary renders", "discordant" in r.summary())

check("kappa identical is 1", close(cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0))
check("kappa opposite is negative", cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0)
# Two labellers each flagging half, independently -> kappa near 0.
check("kappa near zero at chance",
      abs(cohens_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 0.01)

point, blo, bhi = bootstrap_metric(strong, ds.labels, 0.5, n_boot=400, seed=3)
check("bootstrap brackets the point estimate", blo <= point <= bhi,
      f"{blo:.3f} {point:.3f} {bhi:.3f}")
check("bootstrap interval is non-trivial", bhi - blo > 0.0)
# Smaller samples must give wider intervals -- the sample-size lesson, tested.
small = subsample(ds, 60, seed=5)
_, slo, shi = bootstrap_metric(small.scores("judge_strong"), small.labels, 0.5,
                               n_boot=400, seed=3)
check("smaller sample widens the interval", (shi - slo) > (bhi - blo),
      f"n=200 width {bhi - blo:.3f} vs n=60 width {shi - slo:.3f}")

cands_p = [
    Candidate("cheap_bad", f1=0.60, cost_cents_per_100=0.30),
    Candidate("mid", f1=0.79, cost_cents_per_100=0.82),
    Candidate("dear_good", f1=0.87, cost_cents_per_100=3.72),
    Candidate("dominated", f1=0.70, cost_cents_per_100=2.00),
]
front = [c.name for c in pareto_frontier(cands_p)]
check("pareto drops the dominated candidate", "dominated" not in front, str(front))
check("pareto keeps the cheapest", "cheap_bad" in front)
check("pareto keeps the best", "dear_good" in front)
check("pareto is cost-ordered",
      front == sorted(front, key=lambda n: next(
          c.cost_cents_per_100 for c in cands_p if c.name == n)))


# ==========================================================================
# dataset
# ==========================================================================

check("dataset is deterministic",
      make_dataset().scores("judge_strong") == make_dataset().scores("judge_strong"))
check("different seed differs",
      make_dataset(seed=1).scores("judge_strong") != ds.scores("judge_strong"))
counts = ds.mode_counts()
check("every failure has a mode", sum(counts.values()) == sum(ds.labels))
check("modes are long-tailed", max(counts.values()) > min(counts.values()))
check("mode_labels is a subset of labels",
      all(
          not (m == 1 and y == 0)
          for m, y in zip(ds.mode_labels("booking_failed"), ds.labels)
      ))
check("mode_labels sums to the mode count",
      sum(ds.mode_labels("missed_handoff")) == counts["missed_handoff"])

try:
    ds.scores("nope")
    check("unknown judge raises", False)
except KeyError:
    check("unknown judge raises", True)

try:
    ds.mode_labels("nope")
    check("unknown mode raises", False)
except KeyError:
    check("unknown mode raises", True)

check("subsample size", len(subsample(ds, 50, seed=2)) == 50)
check("subsample keeps judges", subsample(ds, 50).judges == ds.judges)

csv_path = Path("/tmp/claude-0/evalkit_test.csv")
ds.to_csv(csv_path)
check("csv written", csv_path.exists() and csv_path.stat().st_size > 1000)
header = csv_path.read_text(encoding="utf-8").splitlines()[0]
check("csv header has judges", "judge_strong" in header and "failure_mode" in header)
csv_path.unlink(missing_ok=True)


# ==========================================================================
# REGRESSIONS -- defects found by adversarial review, 21 Oct 2026
#
# Each of these fails against the first release. They are kept separate so the
# course can point at them: these are not hypothetical failure modes, they are
# the ones that actually got through 112 passing checks.
# ==========================================================================

# REG-C1: isotonic regression must treat tied scores as one input.
# Old behaviour: tied scores entered PAVA as separate one-point blocks, so
# identical inputs could receive different calibrated values and the fit
# depended on the order the data arrived in.
tie_s = [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.9]
tie_y = [0, 1, 0, 1, 1, 0, 1]
iso_t = IsotonicCalibrator().fit(tie_s, tie_y)
out_t = iso_t.transform(tie_s)
check("REG-C1 tied inputs map to one output",
      len({round(v, 12) for s, v in zip(tie_s, out_t) if s == 0.3}) == 1,
      str(out_t))
# And the fit must not depend on presentation order.
perm = [6, 1, 4, 0, 5, 3, 2]
iso_p = IsotonicCalibrator().fit([tie_s[i] for i in perm], [tie_y[i] for i in perm])
check("REG-C1 fit is order-invariant",
      [round(v, 12) for v in iso_t.transform([0.3, 0.7, 0.9])]
      == [round(v, 12) for v in iso_p.transform([0.3, 0.7, 0.9])],
      f"{iso_t.transform([0.3, 0.7, 0.9])} vs {iso_p.transform([0.3, 0.7, 0.9])}")
check("REG-C1 output still monotone on ties",
      all(a <= b + 1e-12 for a, b in
          zip(iso_t.transform([0.3, 0.7, 0.9]), iso_t.transform([0.7, 0.9, 1.0]))))
check("REG-C1 empty fit does not raise",
      IsotonicCalibrator().fit([], []).transform([0.5]) == [0.0])

# REG-C2: paired functions must reject mismatched lengths rather than letting
# zip() silently truncate. Old behaviour: a metric over 180 of 200 labels,
# returning a plausible number with no warning.
_short, _long = [0.1, 0.9], [1, 0, 1]
for _name, _fn in (
    ("brier", lambda: brier(_short, _long)),
    ("log_loss", lambda: log_loss(_short, _long)),
    ("ece", lambda: ece(_short, _long)),
    ("reliability_bins", lambda: reliability_bins(_short, _long)),
    ("brier_decomposition", lambda: brier_decomposition(_short, _long)),
    ("PlattScaler.fit", lambda: PlattScaler().fit(_short, _long)),
    ("IsotonicCalibrator.fit", lambda: IsotonicCalibrator().fit(_short, _long)),
    ("bootstrap_metric", lambda: bootstrap_metric(_short, _long, 0.5, n_boot=5)),
    ("mcnemar", lambda: mcnemar([1, 0], [1, 0], [1, 0, 1])),
    ("agreement", lambda: agreement([1, 0], [1, 0, 1])),
    ("cohens_kappa", lambda: cohens_kappa([1, 0], [1, 0, 1])),
):
    try:
        _fn()
        check(f"REG-C2 {_name} rejects length mismatch", False, "no error raised")
    except ValueError:
        check(f"REG-C2 {_name} rejects length mismatch", True)

# REG-C3: F1 differences need a paired bootstrap on the DIFFERENCE, not
# McNemar (which tests marginal error rates) and not two overlapping
# one-sample intervals.
d_oracle = best_threshold(oracle, ds.labels)
d_weak = best_threshold(weak, ds.labels)
delta = paired_bootstrap_delta(oracle, weak, ds.labels,
                               d_oracle.threshold, d_weak.threshold,
                               n_boot=800, seed=4)
check("REG-C3 delta equals the difference of the two point estimates",
      close(delta.delta, delta.metric_a - delta.metric_b, 1e-9))
check("REG-C3 interval brackets the observed delta",
      delta.lo <= delta.delta <= delta.hi,
      f"{delta.lo:.3f} {delta.delta:.3f} {delta.hi:.3f}")
check("REG-C3 oracle beats weak judge detectably", delta.significant,
      delta.summary())
# A judge against itself must produce a zero delta that is NOT significant.
same = paired_bootstrap_delta(oracle, oracle, ds.labels,
                              d_oracle.threshold, d_oracle.threshold,
                              n_boot=400, seed=4)
check("REG-C3 self-comparison delta is zero", close(same.delta, 0.0, 1e-12))
check("REG-C3 self-comparison is not significant", not same.significant,
      same.summary())
check("REG-C3 rejects length mismatch",
      _try_raises(lambda: paired_bootstrap_delta([0.1], [0.2, 0.3], [1],
                                                 0.5, 0.5, n_boot=5)))
check("REG-C3 summary renders", "delta=" in delta.summary())


# ==========================================================================
# report
# ==========================================================================

def test_all_checks_passed() -> None:
    """Lets pytest run this file correctly.

    The checks above execute at import. Without this sentinel pytest would find
    no test function: a passing suite would report "no tests ran" (exit 5, a
    false red) and a failing one would exit during collection. This turns the
    recorded result into one pytest outcome.
    """
    assert not _FAIL, f"{len(_FAIL)} check(s) failed: " + "; ".join(_FAIL[:10])


if __name__ == "__main__":
    print(f"\n{_PASS} checks passed")
    if _FAIL:
        print(f"{len(_FAIL)} FAILED:")
        for f in _FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print(f"all green ({_PASS} checks)")
