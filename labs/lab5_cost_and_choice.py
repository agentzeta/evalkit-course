"""Lab 5 -- Choosing a judge: floors, then cost, then significance.

Run:  python3 -I labs/lab5_cost_and_choice.py

The order matters and most teams get it backwards. They compare F1 across
models, pick the winner, then discover it costs too much or that the winning
margin was noise. The right order:

  1. Floors      -- disqualify judges that cannot do the job at any threshold.
  2. Cost        -- among survivors, find the Pareto frontier.
  3. Significance-- check the remaining gaps are larger than sampling noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    EXAMPLE_COSTS,
    Candidate,
    bootstrap_metric,
    confusion_at,
    cost_per_month,
    make_dataset,
    mcnemar,
    paired_bootstrap_delta,
    pareto_frontier,
    stratified_folds,
    tune_cv,
)

MIN_P, MIN_R = 0.70, 0.70
RUNS_PER_DAY = 5_000

ds = make_dataset()
labels = ds.labels
folds = stratified_folds(labels, k=5, seed=42)

print(__doc__)
print("=" * 78)
print(f"\nFloors: P >= {MIN_P:.0%}, R >= {MIN_R:.0%}.  "
      f"Assumed volume: {RUNS_PER_DAY:,} eval runs/day\n")

# ------------------------------------------------------- step 1: floors
print("--- Step 1: floors ---\n")
reports, survivors = {}, []
print(f"{'judge':<20} {'held-out F1':>12} {'P':>7} {'R':>7} {'verdict':>14}")
print("-" * 64)
for j in ds.judges:
    rep = tune_cv(ds.scores(j), labels, min_precision=MIN_P, min_recall=MIN_R,
                  folds=folds, model=j)
    reports[j] = rep
    if rep.recommended_threshold is None:
        print(f"{j:<20} {'--':>12} {'--':>7} {'--':>7} {'DISQUALIFIED':>14}")
        continue
    survivors.append(j)
    print(f"{j:<20} {rep.mean_test_f1:>12.3f} {rep.mean_test_precision:>7.3f} "
          f"{rep.mean_test_recall:>7.3f} {'qualifies':>14}")

print(f"\n{len(survivors)} of {len(ds.judges)} judges survive the floors.")
print("Note what just happened: the cheapest judge may already be gone, and no")
print("amount of cost analysis would have told you that.")

# ------------------------------------------------------- step 2: cost
print("\n" + "=" * 78)
print("\n--- Step 2: cost among survivors ---\n")
cands = [
    Candidate(name=j, f1=reports[j].mean_test_f1 or 0.0,
              cost_cents_per_100=EXAMPLE_COSTS[j])
    for j in survivors
]
print(f"{'judge':<20} {'F1':>7} {'c/100 runs':>12} {'$/month':>10} "
      f"{'on frontier':>13}")
print("-" * 68)
front = {c.name for c in pareto_frontier(cands)}
for c in sorted(cands, key=lambda x: x.cost_cents_per_100):
    print(f"{c.name:<20} {c.f1:>7.3f} {c.cost_cents_per_100:>12.2f} "
          f"{cost_per_month(c.cost_cents_per_100, RUNS_PER_DAY):>10.2f} "
          f"{'yes' if c.name in front else 'dominated':>13}")

if len(front) > 1:
    cheap = min((c for c in cands if c.name in front),
                key=lambda c: c.cost_cents_per_100)
    dear = max((c for c in cands if c.name in front), key=lambda c: c.f1)
    if cheap.name != dear.name:
        d_f1 = dear.f1 - cheap.f1
        d_cost = (cost_per_month(dear.cost_cents_per_100, RUNS_PER_DAY)
                  - cost_per_month(cheap.cost_cents_per_100, RUNS_PER_DAY))
        print()
        print(f"The trade on the frontier: +{d_f1:.3f} F1 for "
              f"+${d_cost:.2f}/month.")
        print("Decide that with a number for what a missed failure costs you.")
        print("If you do not have that number, you are not choosing, you are")
        print("guessing with extra steps.")

# ------------------------------------------- step 3: is the gap real?
print("\n" + "=" * 78)
print("\n--- Step 3: is the difference larger than noise? ---\n")

if len(survivors) >= 2:
    a, b = survivors[0], survivors[1]
    ta = reports[a].recommended_threshold
    tb = reports[b].recommended_threshold

    print("USE THE RIGHT TEST FOR THE METRIC YOU ARE QUOTING.\n")
    print("  F1 / precision / recall  ->  paired bootstrap on the DIFFERENCE")
    print("  accuracy / error rate    ->  McNemar\n")
    print("McNemar tests whether two classifiers differ in marginal error")
    print("rate. It is not a test of an F1 difference: F1 is a ratio whose")
    print("denominator changes between the two judges, which the discordant-")
    print("pair count does not capture. Quoting a McNemar p-value beside an F1")
    print("comparison is a category error. (This package shipped that error in")
    print("its first release.)\n")

    print(f"--- F1, the metric we actually chose on: {a} against {b} ---\n")
    delta = paired_bootstrap_delta(ds.scores(a), ds.scores(b), labels, ta, tb,
                                   n_boot=2000, seed=7)
    print(f"  {delta.summary()}")
    print()
    print("  Each bootstrap iteration resamples traces once and recomputes")
    print("  BOTH judges on the same resampled traces, so the shared")
    print("  difficulty of a trace cancels. Comparing two separate one-sample")
    print("  intervals by eye is weaker: overlapping intervals do NOT imply")
    print("  no difference.")
    print()

    print(f"--- Accuracy, where McNemar is the right tool ---\n")
    pred_a = [1 if s >= ta else 0 for s in ds.scores(a)]
    pred_b = [1 if s >= tb else 0 for s in ds.scores(b)]
    res = mcnemar(pred_a, pred_b, labels)
    print(f"  {res.summary()}")
    verdict = ("difference in error rate survives the test" if res.p_value < 0.05
               else "NOT distinguishable on this much data")
    print(f"  p = {res.p_value:.4f}  ->  {verdict}")
    print()
    print(f"  Only {res.n_discordant} traces differ between these two judges.")
    print("  Every other trace they agree on, and agreement carries no")
    print("  information about which is better.")
else:
    print("Fewer than two judges survived; no comparison to make.")
    print("That is a complete and honest result -- report it.")

print()
print("--- And the width of what you are quoting ---\n")
print(f"{'judge':<20} {'F1':>7} {'95% bootstrap interval':>26} {'width':>8}")
print("-" * 64)
for j in survivors:
    t = reports[j].recommended_threshold
    point, lo, hi = bootstrap_metric(ds.scores(j), labels, t, n_boot=2000, seed=5)
    print(f"{j:<20} {point:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>26} "
          f"{hi - lo:>8.3f}")

print()
print("If two judges' intervals overlap substantially, 'we picked the better")
print("one' is not a claim your data supports. Pick on cost, latency or")
print("maintainability instead, and say that is why.")
print()
print("EXERCISE")
print("  1. Put a dollar figure on one missed failure in your own product.")
print("     Now recompute: which judge on the frontier is correct?")
print("  2. Drop the floors to 0.50/0.50. How many judges qualify, and would")
print("     you ship any of them? What does that tell you about floors as a")
print("     tool rather than a formality?")
print("  3. The costs here are illustrative constants. Replace them with your")
print("     provider's real prices and your measured token counts per trace.")
