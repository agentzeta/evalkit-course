"""Tests for design effect and effective sample size under clustering.

Run:  python3 -I tests/test_design.py

The three cases that pin the estimator down are the ones with known answers:

  ICC = 1 by construction (label constant within cluster) must give
    effective n = number of clusters, exactly.
  ICC = 0 by construction (pure noise, no cluster structure) must give
    effective n = number of rows, and must NOT invent clustering.
  A known variance ratio must be recovered: donor variance 1 and cell variance 1
    is a true ICC of 0.5, and the estimate must land near it.

Plus the refusals: a single cluster and singleton clusters carry no information
about the variance they would need, and the honest answer is "not estimable",
not zero.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchaudit import (  # noqa: E402
    CRITICAL,
    OK,
    WARNING,
    as_json,
    as_markdown,
    audit,
    clusters_for_power,
    design_effect,
    from_lists,
    n_for_power,
)

_PASS = 0
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAIL.append(f"{name}: {detail}" if detail else name)


def _try(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


# ===================================================================
# ICC = 1 by construction
# ===================================================================

vals, grps = [], []
for d in range(8):
    for _ in range(60):
        vals.append(float(d % 2))
        grps.append(f"donor{d}")
de = design_effect(vals, grps)

check("constant-within: estimable", de.estimable)
check("constant-within: ICC is 1", abs(de.icc - 1.0) < 1e-9, str(de.icc))
check("constant-within: design effect equals mean cluster size",
      abs(de.design_effect - 60.0) < 1e-6, str(de.design_effect))
check("constant-within: effective n equals the cluster count",
      abs(de.effective_n - 8.0) < 1e-6, str(de.effective_n))
check("constant-within: inflation is sqrt of the design effect",
      abs(de.inflation - math.sqrt(60.0)) < 1e-9)
check("constant-within: rows and clusters recorded",
      de.n_rows == 480 and de.n_clusters == 8)
check("constant-within: mean cluster size recorded", de.mean_cluster == 60.0)
check("constant-within: design is balanced", de.balanced)
check("constant-within: sentence names the inflation factor",
      "too narrow by a factor of" in de.sentence())
check("constant-within: 480 rows are 8 observations, not 480",
      round(de.effective_n) == 8,
      "this is the pseudoreplication claim the tool exists to make")

# ===================================================================
# ICC = 0 by construction — the negative control
# ===================================================================

random.seed(1)
noise = [random.gauss(0, 1) for _ in range(480)]
same_groups = [f"donor{i % 8}" for i in range(480)]
nd = design_effect(noise, same_groups)

check("no structure: estimable", nd.estimable)
check("no structure: ICC is at or near zero", nd.icc < 0.02, str(nd.icc))
check("no structure: design effect is about 1", abs(nd.design_effect - 1.0) < 0.5,
      str(nd.design_effect))
check("no structure: effective n is close to the row count",
      nd.effective_n > 400, str(nd.effective_n))
check("no structure: clustering is not invented", nd.icc >= 0.0,
      "a negative ICC estimate must not become a negative design effect")
check("no structure: clamping is reported, not silent",
      "clamped" in nd.reason or nd.icc > 0.0, nd.reason)
check("no structure: the clamp note says the figure is an upper bound",
      "upper bound" in nd.reason if "clamped" in nd.reason else True)

# ===================================================================
# a known variance ratio must be recovered
# ===================================================================

random.seed(2)
v, g = [], []
for d in range(40):
    donor = random.gauss(0, 1)
    for _ in range(25):
        v.append(donor + random.gauss(0, 1))
        g.append(f"donor{d}")
kd = design_effect(v, g)
check("known ratio: ICC near the true 0.5", abs(kd.icc - 0.5) < 0.12, str(kd.icc))
check("known ratio: effective n is far below the row count",
      kd.effective_n < 150, str(kd.effective_n))
check("known ratio: effective n exceeds the cluster count",
      kd.effective_n > kd.n_clusters,
      "with ICC below 1 the clusters are worth slightly more than one unit each")
check("known ratio: mean squares are reported",
      kd.ms_between > kd.ms_within > 0)

# ICC must rise when between-cluster variance rises.
random.seed(3)
v2, g2 = [], []
for d in range(40):
    donor = random.gauss(0, 3)
    for _ in range(25):
        v2.append(donor + random.gauss(0, 1))
        g2.append(f"donor{d}")
kd2 = design_effect(v2, g2)
check("ICC rises with between-cluster variance", kd2.icc > kd.icc,
      f"{kd2.icc:.3f} vs {kd.icc:.3f}")
check("effective n falls as ICC rises", kd2.effective_n < kd.effective_n)

# ===================================================================
# refusals
# ===================================================================

one = design_effect([1.0, 2.0, 3.0], ["a", "a", "a"])
check("a single cluster is not estimable", not one.estimable)
check("the single-cluster reason is stated",
      "single cluster" in one.reason, one.reason)
check("an unestimable result reports nan, not zero", math.isnan(one.icc),
      "returning 0 would claim the rows are independent")

singles = design_effect([1.0, 2.0, 3.0], ["a", "b", "c"])
check("singleton clusters are not estimable", not singles.estimable)
check("the singleton reason is stated", "one member" in singles.reason)

const = design_effect([5.0] * 20, [f"g{i % 4}" for i in range(20)])
check("a constant measurement is not estimable", not const.estimable)
check("the constant reason is stated", "no variance" in const.reason, const.reason)

check("empty input is not estimable", not design_effect([], []).estimable)
check("ragged input is refused", _try(lambda: design_effect([1.0], ["a", "b"])))
check("nan values are dropped",
      design_effect([1.0, float("nan"), 2.0, 3.0], ["a", "a", "b", "b"]).n_rows == 3)
check("the unestimable sentence explains itself",
      "not estimable" in one.sentence())

# Unbalanced clusters must still produce a figure, using m0 not the plain mean.
unbal_vals, unbal_g = [], []
for d, size in enumerate([2, 5, 40, 100]):
    for _ in range(size):
        unbal_vals.append(float(d))
        unbal_g.append(f"g{d}")
ub = design_effect(unbal_vals, unbal_g)
check("unbalanced clusters are handled", ub.estimable)
check("unbalanced design is flagged as such", not ub.balanced)
check("unbalanced: constant-within still gives ICC 1", abs(ub.icc - 1.0) < 1e-6)

# ===================================================================
# planning arithmetic
# ===================================================================

check("more rows are needed at higher ICC",
      n_for_power(30, 60, 0.5) > n_for_power(30, 60, 0.05))
check("at ICC 0 the rows needed equal the effective n",
      n_for_power(30, 60, 0.0) == 30)
check("clusters needed is rows over cluster size",
      clusters_for_power(30, 60, 0.5) == math.ceil(n_for_power(30, 60, 0.5) / 60))
check("zero target needs nothing", n_for_power(0, 60, 0.5) == 0)
check("zero cluster size does not divide by zero", n_for_power(30, 0, 0.5) == 0)
check("at high ICC, more donors is the only remedy",
      clusters_for_power(30, 60, 0.95) > clusters_for_power(30, 60, 0.05),
      "additional rows within existing clusters buy almost nothing")

# ===================================================================
# integration with the audit
# ===================================================================

random.seed(4)
items, labs, grp = [], [], []
for d in range(8):
    base = d % 2
    for c in range(60):
        flip = base if random.random() < 0.9 else 1 - base
        items.append(f"donor {d} sample {c} readout value {random.randint(100, 999)}")
        labs.append("case" if flip else "ctrl")
        grp.append(f"donor{d}")
res = audit(from_lists(items, labs, grp, name="clustered"))

checks_run = {f.check for f in res.findings}
check("the audit runs the design-effect check", "design-effect" in checks_run)
dfind = next(f for f in res.findings if f.check == "design-effect")
check("a design effect above 10 is CRITICAL", dfind.severity == CRITICAL, dfind.severity)
check("the finding names pseudoreplication", "pseudoreplication" in dfind.detail)
check("the finding says more rows will not help", "more clusters" in dfind.detail)
check("the finding carries the ICC", "icc" in dfind.evidence)
check("the finding carries the inflation factor", "interval_inflation" in dfind.evidence)
check("the grouping column is never blank",
      dfind.evidence["grouping_column"].strip() != "")

check("the headline effective n comes from the declared unit",
      res.effective_n == round(res.design.effective_n),
      f"{res.effective_n} vs {res.design.effective_n}")
check("the source of the effective n is recorded",
      "design effect" in res.effective_n_source)

# A declared grouping must outrank a guess from item text.
mfind = next(f for f in res.findings if f.check == "multiplicity")
check("near-duplicate guessing is demoted when a unit is declared",
      mfind.severity != CRITICAL, mfind.severity)
check("the demotion is explained",
      "authoritative" in mfind.detail, mfind.detail[-200:])
check("the cross-check role is stated", "cross-check" in mfind.detail)
check("the demotion is recorded in evidence",
      mfind.evidence.get("declared_grouping_present") is True)

md = as_markdown(res)
check("markdown shows the design effect", "Design effect" in md)
check("markdown shows the ICC", "| ICC |" in md)
check("markdown shows the interval inflation", "Interval inflation" in md)
check("markdown headlines the declared effective n",
      "from the declared clustering unit" in md)
check("markdown marks the text-based figure as a cross-check",
      "cross-check only" in md)

import json  # noqa: E402

js = json.loads(as_json(res))
check("json carries the design effect block", js["design_effect"] is not None)
check("json design block has the ICC", "icc" in js["design_effect"])
check("json records the effective-n source", "design effect" in js["effective_n_source"])

# No grouping column: the design check must sit out, not guess.
nores = audit(from_lists(items, labs, name="ungrouped"))
check("without a grouping column the design check is skipped",
      "design-effect" not in {f.check for f in nores.findings})
check("without grouping, effective n falls back to clustering",
      "clustering" in nores.effective_n_source)
check("without grouping, json omits the design block",
      json.loads(as_json(nores))["design_effect"] is None)

# A benchmark with a genuine grouping but no real clustering must come back OK.
random.seed(5)
flat_items = [f"independent record {i} with its own distinct content {random.randint(1,10**6)}"
              for i in range(240)]
flat_labs = ["case" if random.random() < 0.5 else "ctrl" for _ in range(240)]
flat_grp = [f"batch{i % 24}" for i in range(240)]
flat = audit(from_lists(flat_items, flat_labs, flat_grp, name="unclustered"))
ff = next((f for f in flat.findings if f.check == "design-effect"), None)
check("a grouping that does not actually cluster is graded OK",
      ff is not None and ff.severity == OK,
      ff.headline if ff else "no design finding")
check("that benchmark's effective n stays near its row count",
      flat.effective_n > 150, str(flat.effective_n))

print(f"\n{_PASS} checks passed")
if _FAIL:
    print(f"{len(_FAIL)} FAILED:")
    for f in _FAIL:
        print(f"  - {f}")
    sys.exit(1)
print(f"all green ({_PASS} checks)")
