"""Lab 7 -- How much data do you actually need?

Run:  python3 -I labs/lab7_sample_size.py

Three questions this lab answers with arithmetic rather than opinion:

  1. How wide is the interval on my headline metric at this sample size?
  2. How much does my tuned threshold move when I resample?
  3. How large an improvement could I even detect?

The third is the one that saves money. If your eval set can only detect a
15-point improvement, running a weekly prompt-tuning cycle that produces
3-point gains is measuring nothing, and every decision made from it is noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    best_threshold,
    bootstrap_metric,
    make_dataset,
    stratified_folds,
    subsample,
    tune_cv,
    wilson_interval,
)

ds = make_dataset(n=1000, seed=4242)  # a bigger pool to draw from
JUDGE = "judge_strong"

print(__doc__)
print("=" * 78)

# ------------------------------------------------- 1: interval widths
print("\n--- 1: what one number costs you, by sample size ---\n")
print(f"{'n traces':>9} {'failures':>9} {'F1':>7} {'95% interval':>22} "
      f"{'width':>8}")
print("-" * 60)
for n in (50, 100, 200, 500, 1000):
    sub = subsample(ds, n, seed=1)
    ch = best_threshold(sub.scores(JUDGE), sub.labels)
    if ch is None:
        continue
    point, lo, hi = bootstrap_metric(sub.scores(JUDGE), sub.labels,
                                     ch.threshold, n_boot=1500, seed=2)
    print(f"{n:>9} {sum(sub.labels):>9} {point:>7.3f} "
          f"{f'[{lo:.3f}, {hi:.3f}]':>22} {hi - lo:>8.3f}")

print()
print("Width shrinks roughly as 1/sqrt(n). Quadrupling your eval set halves the")
print("interval -- which is why 'we added 20 more traces' rarely changes")
print("anything you can see.")

# ------------------------------------------------- 2: threshold stability
print("\n" + "=" * 78)
print("\n--- 2: how much does the tuned threshold wander? ---\n")
print("Same judge, same procedure, different random splits.\n")
print(f"{'n traces':>9} {'thresholds across 6 seeds':>42} {'spread':>8}")
print("-" * 62)
for n in (50, 100, 200, 500, 1000):
    sub = subsample(ds, n, seed=1)
    ts = []
    for seed in range(6):
        try:
            rep = tune_cv(sub.scores(JUDGE), sub.labels, k=5, seed=seed,
                          model=JUDGE)
        except ValueError:
            continue
        if rep.recommended_threshold is not None:
            ts.append(rep.recommended_threshold)
    if not ts:
        continue
    shown = " ".join(f"{t:.3f}" for t in ts)
    print(f"{n:>9} {shown:>42} {max(ts) - min(ts):>8.3f}")

print()
print("The spread is the precision to which you know your threshold. Reporting")
print("'we set the threshold to 0.5174' off 100 traces claims four digits of")
print("knowledge you do not have. Round to the spread.")

# ------------------------------------------------- 3: detectable effect
print("\n" + "=" * 78)
print("\n--- 3: the smallest improvement you could detect ---\n")
print("Two-proportion rule of thumb at 80% power, 5% significance: to detect a")
print("change in recall from p1 to p2 you need roughly")
print()
print("    n per group  ~=  16 * p*(1-p) / (p2-p1)^2       [p = (p1+p2)/2]")
print()
print("Applied around a baseline recall of 75%:\n")
print(f"{'improvement':>13} {'target':>8} {'failures needed':>17} "
      f"{'traces @ 25% base':>19}")
print("-" * 62)
p1 = 0.75
for delta in (0.20, 0.15, 0.10, 0.05, 0.03, 0.02):
    p2 = p1 + delta
    p = (p1 + p2) / 2
    n_fail = 16 * p * (1 - p) / (delta**2)
    print(f"{delta:>12.0%} {p2:>8.0%} {n_fail:>17.0f} {n_fail / 0.25:>19.0f}")

print()
print("Read the bottom row. Detecting a 2-point recall improvement needs about")
print(f"{16 * 0.76 * 0.24 / 0.0004:.0f} labelled failures -- roughly "
      f"{16 * 0.76 * 0.24 / 0.0004 / 0.25:.0f} traces at this base rate.")
print("Nobody has that. So: stop trying to detect 2-point moves. Either chase")
print("larger improvements, or change the unit of analysis -- per-mode judges")
print("on a mode where you have concentrated examples.")
print()
print("--- And the same table as an interval, which is more honest ---\n")
print(f"{'failures':>9} {'observed R=0.75 ->':>20} {'95% interval':>22} "
      f"{'width':>8}")
print("-" * 62)
for n in (20, 50, 100, 200, 500):
    lo, hi = wilson_interval(int(round(0.75 * n)), n)
    print(f"{n:>9} {'':>20} {f'[{lo:.3f}, {hi:.3f}]':>22} {hi - lo:>8.3f}")

print()
print("EXERCISE")
print("  1. Your eval set has 40 labelled failures. Your prompt change moved")
print("     recall from 0.72 to 0.78. Write the one-sentence honest summary.")
print("  2. You need to detect a 5-point recall change. How many failures do")
print("     you need, and how long will it take your product to generate them?")
print("  3. Given (2), argue for or against blocking CI on this metric.")
