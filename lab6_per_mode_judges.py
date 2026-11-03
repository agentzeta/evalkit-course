"""Lab 6 -- What a single aggregate score hides.

Run:  python3 -I labs/lab6_per_mode_judges.py

The usual advice is "build one judge per failure mode, not one judge for 'is
this trace bad'". This lab establishes WHY, and it does so by reporting a result
that contradicts the lazy version of that advice.

Running the lab produces two findings:

  1. A broad judge that scores well in aggregate performs wildly unevenly
     across modes. The headline figure is an average over modes you catch
     almost always and modes you barely catch at all, and it reports neither.

  2. Slicing a GENERIC judge per mode does not fix this. The per-mode figures
     below are worse than the aggregate, not better, because the judge was
     never built to discriminate one mode from another -- it was built to
     separate "bad" from "good". Narrowing the MEASUREMENT does not narrow the
     JUDGE.

So the argument for per-mode judges is not "the numbers go up when you slice".
It is: the aggregate is uninformative about the thing you need to fix, and
fixing it requires a per-mode PROMPT, which is work this synthetic dataset
cannot do for you. What the lab can show is why you must do it.

The third finding is the sample-size wall, and it is the one that constrains
real projects: once you slice 200 traces by mode, most modes have single-digit
example counts and no metric computed on them means anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    FAILURE_MODES,
    best_threshold,
    confusion_at,
    f1,
    make_dataset,
    recall_interval,
    stratified_folds,
    tune_cv,
    wilson_interval,
)

MIN_POSITIVES = 5  # below this, refuse to report a tuned metric at all

ds = make_dataset()
labels = ds.labels

print(__doc__)
print("=" * 82)

# --------------------------------------------------- the broad judge
print("\n--- 1: the broad judge, 'does this trace contain any failure?' ---\n")
folds = stratified_folds(labels, k=5, seed=42)
broad = tune_cv(ds.scores("judge_strong"), labels, min_precision=0.60,
                min_recall=0.60, folds=folds, model="broad")
print(f"held-out F1 {broad.mean_test_f1:.3f}   "
      f"P {broad.mean_test_precision:.3f}   R {broad.mean_test_recall:.3f}   "
      f"threshold {broad.recommended_threshold:.3f}")
print()
print("That is a respectable number and it is about to turn out to mean very")
print("little on its own.")

# --------------------------------------------------- per-mode slicing
print("\n" + "=" * 82)
print("\n--- 2: the same judges, measured one mode at a time ---\n")
print("Each row tunes a threshold against ONE mode's labels, with every other")
print("trace -- including failures of other modes -- counted as a pass. The")
print("judge is unchanged; only the question has narrowed.\n")
print(f"{'failure mode':<24} {'n':>4} {'tuned F1':>9} {'recall':>8} "
      f"{'95% CI on recall':>20} {'status':>12}")
print("-" * 82)

counts = ds.mode_counts()
measurable, per_mode_f1 = [], {}
for mode in FAILURE_MODES:
    y = ds.mode_labels(mode)
    n_pos = sum(y)

    if n_pos < MIN_POSITIVES:
        print(f"{mode:<24} {n_pos:>4} {'--':>9} {'--':>8} {'--':>20} "
              f"{'TOO FEW':>12}")
        continue

    best_j, best_choice = None, None
    for j in ds.judges:
        ch = best_threshold(ds.scores(j), y)
        if ch and (best_choice is None or ch.objective > best_choice.objective):
            best_j, best_choice = j, ch

    c = confusion_at(ds.scores(best_j), y, best_choice.threshold)
    lo, hi = recall_interval(c)
    measurable.append(mode)
    per_mode_f1[mode] = best_choice.objective
    print(f"{mode:<24} {n_pos:>4} {best_choice.objective:>9.3f} "
          f"{best_choice.recall:>8.3f} {f'[{lo:.2f}, {hi:.2f}]':>20} "
          f"{'measurable':>12}")

print()
if per_mode_f1:
    worst = min(per_mode_f1, key=per_mode_f1.get)
    best_m = max(per_mode_f1, key=per_mode_f1.get)
    print(f"Best mode  {best_m} at F1 {per_mode_f1[best_m]:.3f}")
    print(f"Worst mode {worst} at F1 {per_mode_f1[worst]:.3f}")
    print(f"Spread     {per_mode_f1[best_m] - per_mode_f1[worst]:.3f}")
    print()
    print(f"Every per-mode figure is BELOW the aggregate of "
          f"{broad.mean_test_f1:.3f}, and these are in-sample numbers, which")
    print("flatter them. A judge trained to separate bad from good does not")
    print("separate one kind of bad from another. If you need to know which")
    print("mode is failing -- and you do, because that is what you fix -- the")
    print("prompt has to ask about that mode.")

print()
print("Now read the confidence intervals. On a mode with five or six examples")
print("the interval spans half the range. Those rows are not measurements, they")
print("are rumours with a decimal point.")

# ------------------------------------------------ what the aggregate hides
print("\n" + "=" * 82)
print("\n--- 3: which modes the broad judge actually misses ---\n")
t = broad.recommended_threshold
print(f"At the broad judge's tuned threshold of {t:.3f}:\n")
print(f"{'failure mode':<24} {'in data':>8} {'missed':>8} {'miss rate':>11}")
print("-" * 54)
for mode in FAILURE_MODES:
    in_data = counts[mode]
    if not in_data:
        continue
    missed = sum(
        1 for tr in ds.traces
        if tr.failure_mode == mode and tr.scores["judge_strong"] < t
    )
    print(f"{mode:<24} {in_data:>8} {missed:>8} {missed / in_data:>10.0%}")

print()
print("The single F1 figure averages all of this. Suppose the mode it misses")
print("most were pii_in_response rather than booking_failed: the headline would")
print("not move, and your worst exposure would be invisible. The aggregate is")
print("not wrong -- it is answering a question nobody at your company asked.")

# ------------------------------------------------ the sample-size wall
print("\n" + "=" * 82)
print("\n--- 4: the sample-size wall ---\n")
n_skipped = len(FAILURE_MODES) - len(measurable)
print(f"{len(measurable)} of {len(FAILURE_MODES)} modes have at least "
      f"{MIN_POSITIVES} examples. {n_skipped} refused outright.")
print()
print("Interval width on recall at an observed 80%, by number of labelled")
print("examples of that mode:\n")
print(f"{'examples':>10} {'95% CI on recall':>22} {'width':>8} {'verdict':>22}")
print("-" * 66)
for n in (5, 10, 20, 50, 100, 200, 400):
    lo, hi = wilson_interval(int(round(0.8 * n)), n)
    w = hi - lo
    verdict = ("useless" if w > 0.4 else "indicative" if w > 0.2
               else "usable" if w > 0.1 else "solid")
    print(f"{n:>10} {f'[{lo:.3f}, {hi:.3f}]':>22} {w:>8.3f} {verdict:>22}")

print()
print("To measure one mode properly you need on the order of 50-100 labelled")
print("examples OF THAT MODE. At a 3% incidence that is 1,500-3,000 traces")
print("reviewed, for one mode. This is the real cost of per-mode evals, it is")
print("almost never budgeted, and it is why sampling strategy -- deliberately")
print("over-sampling rare modes rather than sampling traffic uniformly -- is")
print("the highest-leverage decision in the whole programme.")
print()
print("EXERCISE")
print("  1. Take the mode the broad judge misses most. Read its snippets from")
print("     Lab 1. Why is it structurally hard for a generic prompt to catch?")
print("  2. Write a judge prompt for that one mode: one paragraph, specific")
print("     enough that two colleagues labelling 20 traces would agree.")
print("  3. You can review 200 traces a week. Design a 6-week sampling plan")
print("     that gets 'repeated_loop' to a recall interval under 0.20. State")
print("     what you will NOT measure as a result of that choice.")
