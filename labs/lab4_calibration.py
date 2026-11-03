"""Lab 4 -- Can you read the probability literally?

Run:  python3 -I labs/lab4_calibration.py

Discrimination and calibration are different properties. This lab builds a
judge that discriminates perfectly and is badly calibrated, to make the
distinction concrete, then shows that fixing calibration is an empirical
question with a measurable answer -- sometimes "no".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    IsotonicCalibrator,
    PlattScaler,
    best_threshold,
    brier,
    brier_decomposition,
    ece,
    evaluate_calibrator,
    log_loss,
    make_dataset,
    reliability_bins,
    stratified_folds,
)
from evalkit.plot import reliability_diagram

OUT = Path(__file__).resolve().parent.parent / "out"
ds = make_dataset()
labels = ds.labels
folds = stratified_folds(labels, k=5, seed=42)

print(__doc__)
print("=" * 78)

# ---------------------------------------------------------------- part 1
print("\n--- Part 1: the same ranking, three different sets of probabilities ---\n")
base = ds.scores("judge_strong")
squashed = [s * 0.45 for s in base]           # systematically under-confident
inflated = [min(0.999, s ** 0.35) for s in base]  # systematically over-confident

print("Each column below has the IDENTICAL ranking of traces, so each gives the")
print("identical F1 once its threshold is tuned. Only the numbers differ.\n")
print(f"{'variant':<14} {'tuned F1':>9} {'Brier':>8} {'log loss':>9} {'ECE':>7}")
print("-" * 52)
for name, s in (("raw", base), ("squashed", squashed), ("inflated", inflated)):
    b = best_threshold(s, labels)
    print(f"{name:<14} {(b.objective if b else 0):>9.3f} "
          f"{brier(s, labels):>8.4f} {log_loss(s, labels):>9.4f} "
          f"{ece(s, labels):>7.4f}")

print()
print("Identical F1, wildly different Brier. If you only threshold the score,")
print("calibration is irrelevant -- the threshold absorbs the distortion. The")
print("moment you route on the number ('auto-close below 0.1'), it is not.")

# ---------------------------------------------------------------- part 2
print("\n" + "=" * 78)
print("\n--- Part 2: reliability, bin by bin ---\n")
print("judge_strong, five equal-count bins:\n")
print(f"{'bin':>4} {'n':>4} {'score range':>16} {'mean pred':>10} "
      f"{'observed':>9} {'gap':>8}")
print("-" * 56)
for i, b in enumerate(reliability_bins(squashed, labels, 5)):
    print(f"{i:>4} {b.n:>4} {b.lo:>7.3f}-{b.hi:<8.3f} {b.mean_pred:>10.3f} "
          f"{b.observed_rate:>9.3f} {b.gap:>+8.3f}")
print()
print("Positive gap = the model under-predicts failure in that bin. The")
print("squashed judge under-predicts everywhere, which is what a systematic")
print("scaling error looks like on a reliability diagram.")

# ---------------------------------------------------------------- part 3
print("\n" + "=" * 78)
print("\n--- Part 3: Murphy's decomposition ---\n")
print("Brier = reliability - resolution + uncertainty\n")
print(f"{'variant':<14} {'Brier':>8} {'reliability':>12} {'resolution':>11} "
      f"{'uncertainty':>12}")
print("-" * 62)
for name, s in (("raw", base), ("squashed", squashed), ("weak judge",
                                                        ds.scores("judge_weak"))):
    rel, res, unc = brier_decomposition(s, labels, 5)
    print(f"{name:<14} {brier(s, labels):>8.4f} {rel:>12.4f} {res:>11.4f} "
          f"{unc:>12.4f}")
print()
print("reliability  fixable by calibration (lower is better)")
print("resolution   real information in the judge (higher is better);")
print("             post-hoc calibration CANNOT increase it")
print("uncertainty  a property of your data; identical in every row")
print()
print("Read the weak judge's resolution. Calibrating it would produce")
print("well-dressed ignorance: honest probabilities that are all near the base")
print("rate, and no more useful for finding failures than a coin.")

# ---------------------------------------------------------------- part 4
print("\n" + "=" * 78)
print("\n--- Part 4: does calibration actually help? (cross-validated) ---\n")
print("Fit the calibrator on training folds, measure on held-out folds. Fitting")
print("and measuring on the same traces always 'improves' things.\n")
print("judge_oracle is the control. It reports the latent probability the")
print("labels were actually drawn from, so it is calibrated BY CONSTRUCTION --")
print("its reliability term is ~0.0003. There is nothing for a calibrator to")
print("fix, so any 'improvement' on it would be overfitting. Watch it fail.\n")

for name, s in (
    ("squashed", squashed),
    ("inflated", inflated),
    ("judge_strong", base),
    ("judge_oracle", ds.scores("judge_oracle")),
    ("judge_weak", ds.scores("judge_weak")),
):
    r_platt = evaluate_calibrator(s, labels, folds, PlattScaler, f"{name} +platt")
    r_iso = evaluate_calibrator(s, labels, folds, IsotonicCalibrator, f"{name} +iso")
    print(r_platt.summary())
    print(r_iso.summary())
    print()

print("Note any row where the verdict is 'NOT improved'. Platt scaling is a")
print("two-parameter sigmoid; on a judge that is already calibrated it can only")
print("add noise. Isotonic is more flexible and therefore overfits harder on")
print("200 traces. Neither is the right answer a priori -- you measure.")

OUT.mkdir(exist_ok=True)
p = PlattScaler().fit(squashed, labels)
reliability_diagram(squashed, labels, OUT / "lab4_reliability.svg", n_bins=5,
                    title="squashed judge: raw against Platt-calibrated",
                    calibrated=p.transform(squashed))
print(f"\nWrote {OUT / 'lab4_reliability.svg'}")
print()
print("EXERCISE")
print("  1. One judge above gets WORSE under calibration. Which, and why?")
print("  2. Build a judge that is perfectly calibrated and useless: return the")
print("     base rate for every trace. What is its Brier? Its resolution? Its")
print("     recall at any threshold?")
print("  3. You want to auto-resolve traces the judge is confident are passes.")
print("     Using the calibrated scores, find the threshold below which the")
print("     observed failure rate is under 2%. How many traces does that")
print("     auto-resolve, and how many failures does it let through?")
