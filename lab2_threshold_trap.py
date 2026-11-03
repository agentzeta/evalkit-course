"""Lab 2 -- The threshold trap.

Run:  python3 -I labs/lab2_threshold_trap.py

The point of this lab is a single comparison. One judge, one set of scores, two
thresholds: the API default of 0.50, and one chosen from the data. Nothing
about the model changes between the two rows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (
    best_threshold,
    confusion_at,
    f1,
    make_dataset,
    metric_row,
    precision,
    recall,
    sweep,
)
from evalkit.plot import pr_vs_threshold, score_distributions

OUT = Path(__file__).resolve().parent.parent / "out"

ds = make_dataset()
scores = ds.scores("judge_shifted")
labels = ds.labels

print(__doc__)
print(f"dataset           {len(ds)} traces, {sum(labels)} failures "
      f"({ds.base_rate:.1%} base rate)")
print(f"score range       {min(scores):.3f} to {max(scores):.3f}")
print()
print("Note the score range. This judge never emits a score above 0.5, which is")
print("not pathological -- plenty of real judges are poorly scaled. Watch what")
print("the default threshold does to it.")
print()

rows = []
c_default = confusion_at(scores, labels, 0.50)
rows.append(("0.500 (API default)", c_default))

tuned = best_threshold(scores, labels, beta=1.0)
assert tuned is not None
rows.append((f"{tuned.threshold:.3f} (tuned for F1)", c_default.__class__(
    **{k: getattr(tuned.row.confusion, k) for k in ("tp", "fp", "fn", "tn")})))

print(f"{'threshold':<22} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
      f"{'prec':>7} {'recall':>7} {'F1':>7}")
print("-" * 72)
for name, c in rows:
    print(f"{name:<22} {c.tp:>4} {c.fp:>4} {c.fn:>4} {c.tn:>4} "
          f"{precision(c):>7.3f} {recall(c):>7.3f} {f1(c):>7.3f}")

print()
print("The default threshold flags nothing, so it catches none of the "
      f"{c_default.n_positive} failures.")
print("The judge was never the problem. The threshold was.")
print()

# Now the constrained version: suppose the alerting system needs 80/80.
constrained = best_threshold(scores, labels, min_precision=0.80, min_recall=0.80)
if constrained is None:
    print("At floors of 80% precision AND 80% recall: no threshold qualifies.")
    print("That is a finding about this judge, not a reason to lower the floors.")
else:
    print(f"At floors of 80/80: threshold {constrained.threshold:.3f} "
          f"gives P={constrained.precision:.3f} R={constrained.recall:.3f}")

print()
print("Compare every judge at the default, to see that this is not a one-off:")
print()
print(f"{'judge':<20} {'max score':>10} {'recall @ 0.50':>14} {'best F1':>9} "
      f"{'at threshold':>13}")
print("-" * 70)
for j in ds.judges:
    s = ds.scores(j)
    r_def = recall(confusion_at(s, labels, 0.50))
    b = best_threshold(s, labels)
    print(f"{j:<20} {max(s):>10.3f} {r_def:>14.3f} "
          f"{(b.objective if b else 0):>9.3f} {(b.threshold if b else 0):>13.3f}")

OUT.mkdir(exist_ok=True)
score_distributions(scores, labels, tuned.threshold,
                    OUT / "lab2_distributions.svg",
                    title="judge_shifted: score distributions with tuned threshold")
pr_vs_threshold(sweep(scores, labels), OUT / "lab2_pr_curve.svg",
                min_precision=0.80, min_recall=0.80,
                title="judge_shifted: precision and recall against threshold")
print()
print(f"Wrote {OUT / 'lab2_distributions.svg'}")
print(f"Wrote {OUT / 'lab2_pr_curve.svg'}")
print()
print("EXERCISE")
print("  1. Change beta to 2.0 (recall matters four times as much as")
print("     precision). Which way does the threshold move, and why?")
print("  2. Set min_recall=0.95 and min_precision=0.50. Does any judge")
print("     qualify? Which one would you ship for an alerting system?")
print("  3. Open lab2_distributions.svg. Where would you put the threshold")
print("     by eye, before computing anything? Were you close?")
