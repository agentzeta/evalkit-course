"""Lab 3 -- Tuning a threshold without lying to yourself.

Run:  python3 -I labs/lab3_tune_cv.py

Two things happen here.

First, the leakage demonstration: a threshold tuned on all the data and scored
on all the data, against the same procedure done under cross-validation. The
gap between those two numbers is the amount by which you would have overstated
your judge if you had skipped the split.

Second, the refusal: a judge that cannot meet the floors produces an empty
recommendation and says which folds failed, rather than quietly relaxing the
requirement to produce a number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    best_threshold,
    bootstrap_metric,
    confusion_at,
    f1,
    make_dataset,
    stratified_folds,
    tune_cv,
)

MIN_P, MIN_R = 0.80, 0.80

ds = make_dataset()
labels = ds.labels

# One set of folds, shared by every judge. Different splits per judge would
# make the comparison meaningless.
folds = stratified_folds(labels, k=5, seed=42)

print(__doc__)
print("=" * 78)
print(f"\nFloors set BEFORE tuning: precision >= {MIN_P:.0%}, "
      f"recall >= {MIN_R:.0%}")
print(f"Folds: 5, stratified, shared across all judges\n")

print(f"{'judge':<20} {'in-sample F1':>13} {'held-out F1':>12} "
      f"{'optimism':>9} {'thresh':>8} {'folds ok':>9}")
print("-" * 78)

results = {}
for j in ds.judges:
    s = ds.scores(j)
    in_sample = best_threshold(s, labels, min_precision=MIN_P, min_recall=MIN_R)
    rep = tune_cv(s, labels, min_precision=MIN_P, min_recall=MIN_R,
                  folds=folds, model=j)
    results[j] = rep

    if in_sample is None and rep.recommended_threshold is None:
        print(f"{j:<20} {'--':>13} {'--':>12} {'--':>9} {'--':>8} "
              f"{0}/{len(rep.folds):>7}")
        continue

    ins = in_sample.objective if in_sample else float("nan")
    out = rep.mean_test_f1 or float("nan")
    thr = rep.recommended_threshold
    n_ok = len(rep.folds) - rep.n_unqualified_folds
    print(f"{j:<20} {ins:>13.3f} {out:>12.3f} {ins - out:>+9.3f} "
          f"{(thr if thr is not None else 0):>8.3f} "
          f"{n_ok}/{len(rep.folds):>7}")

print()
print("The 'optimism' column is the cost of skipping the split. Positive means")
print("the in-sample number flattered the judge. Report the held-out column.")
print()

print("=" * 78)
print("\n--- The refusal, in full ---\n")
hard = tune_cv(ds.scores("judge_weak"), labels,
               min_precision=0.95, min_recall=0.95, folds=folds,
               model="judge_weak")
print(hard.summary())
print()
print("Five folds, no qualifying threshold in any of them. The library returns")
print("None rather than a number. If your tooling instead reports the best")
print("available F1 here, it has answered a question you did not ask.")
print()

print("=" * 78)
print("\n--- Per-fold detail for the judge you would actually ship ---\n")
best_judge = max(
    (j for j in ds.judges if results[j].recommended_threshold is not None),
    key=lambda j: results[j].mean_test_f1 or 0,
)
rep = results[best_judge]
print(f"judge: {best_judge}\n")
print(f"{'fold':>5} {'threshold':>10} {'test P':>8} {'test R':>8} "
      f"{'test F1':>8} {'meets floors':>14}")
print("-" * 58)
for f in rep.folds:
    if f.test_row is None:
        print(f"{f.fold:>5} {'none':>10} {'--':>8} {'--':>8} {'--':>8} "
              f"{'no threshold':>14}")
        continue
    print(f"{f.fold:>5} {f.threshold:>10.3f} {f.test_row.precision:>8.3f} "
          f"{f.test_row.recall:>8.3f} {f.test_row.f1:>8.3f} "
          f"{'yes' if f.meets_floors_on_test else 'NO':>14}")
print()
print(rep.summary())
print()
print(f"Folds that met the floors on training data but missed them on held-out")
print(f"data: {rep.n_failed_on_test}. That number is your overfitting, measured.")
print()

thr = rep.recommended_threshold
point, lo, hi = bootstrap_metric(ds.scores(best_judge), labels, thr,
                                 n_boot=2000, seed=11)
print(f"F1 at the recommended threshold, with a 95% bootstrap interval:")
print(f"  {point:.3f}  [{lo:.3f}, {hi:.3f}]   width {hi - lo:.3f}")
print()
print("Quote the interval, not the point. A rival judge inside this interval is")
print("not worse than yours; it is indistinguishable on this much data.")
print()
print("EXERCISE")
print("  1. Re-run with seed=7 and seed=99 in stratified_folds. How much does")
print("     the recommended threshold move? That movement is the precision to")
print("     which you actually know the threshold -- do not quote more digits.")
print("  2. Raise the floors to 0.90/0.90. Which judges survive?")
print("  3. Set beta=2.0 and re-run. Explain to a colleague, in one sentence,")
print("     what business situation justifies that choice.")
