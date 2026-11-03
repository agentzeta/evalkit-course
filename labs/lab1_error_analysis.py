"""Lab 1 -- Error analysis: look at your data.

Run:  python3 -I labs/lab1_error_analysis.py [n]

This is the lab people skip and the one that matters most. Before any metric,
any judge, any threshold: read traces and write down what went wrong, in your
own words, without a predefined list.

The method is open coding then axial coding, borrowed from qualitative social
science. Open coding: read each failing trace, write a short free-text note on
what went wrong. Axial coding: group your notes into categories, merging and
splitting until each category is something you could write a rule for. Those
categories become your failure modes, and your judges are built one per mode.

Do NOT start from a vendor's taxonomy of "hallucination / toxicity / relevance".
Your application's failures are specific to your application. In this dataset
the real modes include "missed handoff to a human on a legal dispute", which no
generic taxonomy contains and which is the most expensive failure here.
"""

import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import FAILURE_MODES, make_dataset

n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 8

ds = make_dataset()
print(__doc__)
print("=" * 74)
print(f"\n{len(ds)} traces, {sum(ds.labels)} failures, "
      f"{ds.base_rate:.1%} base rate\n")

failing = [t for t in ds.traces if t.label == 1]
print(f"--- {n_show} failing traces. Read them. Write your own notes first. ---\n")
for t in failing[:n_show]:
    print(f"[{t.trace_id}]")
    for line in textwrap.wrap(t.text, 70):
        print(f"  {line}")
    print()

input_hint = (
    "Cover the next section before showing this to a class. The exercise is\n"
    "worthless if they see the answer key first.\n"
)
print("=" * 74)
print(input_hint)
print("=" * 74)
print("\n--- Reference taxonomy and frequencies ---\n")

counts = Counter(t.failure_mode for t in failing)
total = sum(counts.values())
print(f"{'failure mode':<24} {'count':>6} {'share of failures':>18} "
      f"{'share of all traces':>20}")
print("-" * 72)
for mode in FAILURE_MODES:
    c = counts.get(mode, 0)
    print(f"{mode:<24} {c:>6} {c / total:>17.1%} {c / len(ds):>19.1%}")

print()
print("Read the bottom of that table carefully.")
print()
rare = min(FAILURE_MODES, key=lambda m: counts.get(m, 0))
rare_n = counts.get(rare, 0)
print(f"  '{rare}' appears {rare_n} times in {len(ds)} traces.")
print(f"  On a 100-trace eval set you would expect about {rare_n / 2:.0f}.")
print("  You cannot measure a judge for it. Any recall figure you quote would")
print("  move in steps of 50% or more. The correct response is to say so and")
print("  go and collect more traces of that mode -- not to report a number.")
print()
print("This is why a single 'overall quality' score is a mistake: it averages a")
print("mode you can measure with one you cannot, and hides which is which.")
print()
print("EXERCISE")
print("  1. Open code 20 failing traces yourself, free-text, no categories.")
print("     Then group your notes. How many modes did you find? Did you split")
print("     anything the reference table merges, or merge anything it splits?")
print("  2. For each mode, write one sentence a second person could apply to")
print("     an unseen trace to get the same answer. If you cannot, the mode is")
print("     not yet a mode -- it is a feeling.")
print("  3. Rank the modes by cost to the business, not by frequency. Which")
print("     mode would you build the first judge for? (In this dataset the")
print("     answer is probably not the most common one.)")
