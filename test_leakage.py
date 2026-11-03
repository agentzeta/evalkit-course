"""Tests for the leakage diagnostics.

Run:  python3 -I tests/test_leakage.py

The important tests here are the NEGATIVE ones. A detector that fires on
everything is useless, so the suite includes a deliberately clean benchmark
and checks that the battery stays quiet on it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import (  # noqa: E402
    ceiling_violation,
    degenerate_baselines,
    group_folds,
    grouping_gap,
    make_dataset,
    naive_bayes_fit_predict,
    row_folds,
    template_multiplicity,
)
from evalkit.thresholds import best_threshold  # noqa: E402

_PASS = 0
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAIL.append(f"{name}: {detail}" if detail else name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _try(fn, exc=ValueError) -> bool:
    """True when fn() raises the expected exception."""
    try:
        fn()
        return False
    except exc:
        return True


# ==========================================================================
# A CLEAN benchmark: items carry no label information at all.
# ==========================================================================

rng = random.Random(7)
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
clean_texts, clean_labels = [], []
for i in range(400):
    # Text is generated INDEPENDENTLY of the label, and every item is unique.
    body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 14)))
    clean_texts.append(f"record {i:04d} {body}")
    clean_labels.append(1 if rng.random() < 0.25 else 0)

clean = ceiling_violation(clean_texts, clean_labels, claimed_ceiling=0.874)
check("clean: every item distinct",
      clean.multiplicity.n_distinct == 400, str(clean.multiplicity.n_distinct))
check("clean: effective n equals row count",
      clean.multiplicity.effective_n == 400)
check("clean: no ceiling violation", not clean.leaks, clean.summary())
# THE negative control: with unique items, nothing is ever seen twice, so the
# out-of-fold identity probe must collapse to the trivial baseline or below.
# The in-sample version of this baseline returned 1.000 here -- a false
# positive on a benchmark with no leakage at all.
check("clean: out-of-fold item identity collapses",
      next(b.f1 for b in clean.baselines
           if b.name.startswith("item identity")) <= 0.45,
      str([str(b) for b in clean.baselines]))
check("clean: verdict says sound-not-complete",
      "sound, not complete" in clean.summary())

# The detector must also stay quiet with no ceiling supplied.
no_ceiling = ceiling_violation(clean_texts, clean_labels)
check("no ceiling: runs diagnostics only", not no_ceiling.leaks)
check("no ceiling: says so", "no ceiling supplied" in no_ceiling.summary())


# ==========================================================================
# A LEAKY benchmark: the package's own teaching fixture.
# ==========================================================================

ds = make_dataset()
L = ds.labels
texts = [t.text for t in ds.traces]
ceiling = best_threshold(ds.scores("judge_oracle"), L).objective

leaky = ceiling_violation(texts, L, claimed_ceiling=ceiling)

check("leaky: 18 distinct items over 200 rows",
      leaky.multiplicity.n_distinct == 18 and leaky.multiplicity.n_rows == 200,
      f"{leaky.multiplicity.n_distinct}/{leaky.multiplicity.n_rows}")
check("leaky: every group has a constant label",
      leaky.multiplicity.all_groups_constant)
check("leaky: effective n collapses to the group count",
      leaky.multiplicity.effective_n == 18)
check("leaky: out-of-fold item identity is near perfect",
      next(b.f1 for b in leaky.baselines
           if b.name.startswith("item identity")) > 0.95,
      str([str(b) for b in leaky.baselines]))
check("leaky: violation detected", leaky.leaks)
check("leaky: verdict names the proof", "LEAKAGE PROVEN" in leaky.summary())
check("leaky: claimed ceiling is the pi-reference score",
      0.80 < ceiling < 0.95, f"{ceiling:.4f}")
check("leaky: violation margin is positive",
      max(b.f1 for b in leaky.violations) - ceiling > 0.05,
      f"margin={max(b.f1 for b in leaky.violations) - ceiling:.4f}")

# Baselines are sorted strongest first, and the trivial one is present.
check("baselines sorted by F1 descending",
      all(a.f1 >= b.f1 for a, b in zip(leaky.baselines, leaky.baselines[1:])))
check("flag-everything baseline present",
      any(b.name == "flag-everything" for b in leaky.baselines))
# F1 of flag-everything at base rate b is 2b/(1+b).
br = ds.base_rate
check("flag-everything F1 matches 2b/(1+b)",
      close(next(b.f1 for b in leaky.baselines if b.name == "flag-everything"),
            2 * br / (1 + br), 1e-9))


# ==========================================================================
# Multiplicity arithmetic
# ==========================================================================

m = template_multiplicity(["a", "a", "a", "b", "b", "c"], [1, 1, 1, 0, 0, 0])
check("multiplicity counts groups", m.n_distinct == 3 and m.n_rows == 6)
check("multiplicity max reuse", m.max_reuse == 3)
check("multiplicity all constant", m.all_groups_constant)
check("multiplicity effective n is the group count", m.effective_n == 3)
# Size-weighted mean cluster size = (9 + 4 + 1) / 6 = 2.333...
check("size-weighted mean cluster", close(m.mean_cluster, 14 / 6, 1e-9),
      f"{m.mean_cluster}")
check("design effect at icc=1 equals group count approx",
      close(m.design_effect_n(1.0), 6 / (14 / 6), 1e-9))
check("design effect at icc=0 equals row count", close(m.design_effect_n(0.0), 6.0))
check("design effect rejects bad icc", _try(lambda: m.design_effect_n(1.5)))

mixed = template_multiplicity(["a", "a", "b"], [1, 0, 1])
check("mixed-label group detected", mixed.mixed_label_groups == 1)
check("mixed groups keep effective n at row count", mixed.effective_n == 3)

check("multiplicity rejects length mismatch", _try(lambda: template_multiplicity(["a"], [1, 0])))
check("multiplicity handles empty", template_multiplicity([], []).n_rows == 0)
check("baselines reject length mismatch", _try(lambda: degenerate_baselines(["a"], [1, 0])))
check("baselines handle empty", degenerate_baselines([], []) == [])


# ==========================================================================
# Splitting
# ==========================================================================

groups = texts
gf = group_folds(groups, k=5, seed=1)
check("group folds partition the rows",
      sorted(i for _, te in gf for i in te) == list(range(200)))
check("group folds never split a group",
      all(
          not ({groups[i] for i in tr} & {groups[i] for i in te})
          for tr, te in gf
      ))
check("group folds refuse when too few groups",
      _try(lambda: group_folds(["a", "b"], k=5)))
check("group folds reject k<2", _try(lambda: group_folds(groups, k=1)))

rf = row_folds(200, 5, 1)
check("row folds partition the rows",
      sorted(i for _, te in rf for i in te) == list(range(200)))

gap = grouping_gap(texts, L, groups, naive_bayes_fit_predict, k=5, seed=1)
check("row-wise CV is near perfect on a leaky fixture",
      gap.row_wise_f1 > 0.95, f"{gap.row_wise_f1:.3f}")
check("grouped CV is substantially lower",
      gap.grouped_f1 < gap.row_wise_f1 - 0.2,
      f"row {gap.row_wise_f1:.3f} grouped {gap.grouped_f1:.3f}")
check("protocol gap is large and positive", gap.gap > 0.3, f"{gap.gap:.3f}")
check("row-wise overlap is near total", gap.overlap_share > 0.8,
      f"{gap.overlap_share:.2f}")
check("grouped folds are unstable at 18 groups",
      max(gap.grouped_folds) - min(gap.grouped_folds) > 0.5,
      str([round(x, 3) for x in gap.grouped_folds]))
check("gap summary renders", "protocol gap" in gap.summary())
check("grouping_gap rejects ragged input",
      _try(lambda: grouping_gap(["a"], [1, 0], ["a"], naive_bayes_fit_predict)))

# On the CLEAN benchmark the two protocols should broadly agree, because
# there is nothing to memorise. This is the negative control for the gap.
clean_gap = grouping_gap(clean_texts, clean_labels, clean_texts,
                         naive_bayes_fit_predict, k=5, seed=1)
check("clean: protocol gap is small",
      abs(clean_gap.gap) < 0.25,
      f"row {clean_gap.row_wise_f1:.3f} grouped {clean_gap.grouped_f1:.3f}")

check("naive bayes returns one prediction per test row",
      len(naive_bayes_fit_predict(texts[:100], L[:100], texts[100:])) == 100)


print(f"\n{_PASS} checks passed")
if _FAIL:
    print(f"{len(_FAIL)} FAILED:")
    for f in _FAIL:
        print(f"  - {f}")
    sys.exit(1)
print(f"all green ({_PASS} checks)")
