# evalkit

**A judge scoring F1 0.000 at the default threshold scores 0.860 tuned. Same judge, same scores.**

A teaching library and lab course for LLM and agent evaluation. Pure Python
standard library — no installation, no dependencies, nothing to break on a
workshop laptop five minutes before you start.

```bash
git clone https://github.com/<you>/evalkit && cd evalkit
python3 -I tests/test_evalkit.py          # 134 checks
python3 -I labs/lab1_error_analysis.py

for t in tests/test_*.py; do python3 -I "$t"; done   # 652 checks in total
```

Every lab is self-contained and prints its own teaching commentary, so a student
can work through them unattended and an instructor can project them.

---

## The labs

| Lab | Subject | The thing it proves |
|---|---|---|
| `lab1_error_analysis.py` | Open and axial coding of traces | Your failure taxonomy comes from your data, and the rarest mode is unmeasurable |
| `lab2_threshold_trap.py` | Decision thresholds | A judge scoring **F1 0.000** at the 0.50 default scores **0.860** tuned |
| `lab3_tune_cv.py` | Honest tuning | In-sample F1 overstates held-out F1; a judge that cannot meet the floors returns *nothing* |
| `lab4_calibration.py` | Probability quality | Three variants with **identical F1** and Brier from 0.069 to 0.158; calibration makes two judges **worse** |
| `lab5_cost_and_choice.py` | Choosing a judge | Floors first, then cost, then significance — and the right test per metric |
| `lab6_per_mode_judges.py` | Scope of an evaluator | An aggregate F1 of 0.837 hides per-mode F1 from 0.217 to 0.667 |
| `lab7_sample_size.py` | How much data | Detecting a 2-point recall gain needs ~7,300 labelled failures |
| `lab8_infrastructure.py` | Eval infrastructure | n=100 at 80% cannot certify `pass^5 ≥ 0.30`; 12 CI categories at α=0.05 is a 46% false-red rate |

---

## The method, in one line

Every result in this repo has the same shape. There is a number everyone
reports, and a second number that decides whether the first one means anything:

| Reported | Decides whether it means anything |
|---|---|
| benchmark score | the trivial baseline `2p/(1+p)` |
| judge F1 | the threshold it was measured at |
| agent success rate | the step budget, and cost per successful task |
| AURC | the AURC of shuffled scores |
| assertion pass rate | taxonomy coverage |
| per-category CI gate | the family-wise false-failure rate and the MDE |
| recall@k | the arithmetic ceiling `k/R` |
| a reported Sharpe ratio | the number of configurations tried |
| pass@k | pass^k |

**The second column is always a baseline, a ceiling, or a coverage figure.** Not
a better metric — a companion number answering *"compared to what?"* or *"out of
what?"*

---

## Three results worth the clone on their own

### `pass^k` is easy to compute and hard to certify

```
n=100 runs, c=80 successes, k=5
  per-run p          0.800  [0.711, 0.867]
  pass@5 (capability) 1.000
  pass^5 (reliability) 0.319  [0.182, 0.489]
  plug-in p^5        0.328   (biased upward when p is estimated)
  error amplification  2.05x
  target pass^5 >= 0.300: NOT CERTIFIED
    point estimate 0.319 is above target, but the lower bound 0.182 is not.
    certification needs the lower bound on p above 0.78600; it is 0.71117.
```

`pass^k` amplifies the error in your estimate of `p` by `k·p^(k-1)`, so the
sample size that was fine for `pass@1` is not fine here. The usual report of
this system is *"pass^5 = 0.33, target 0.30, ship it."*

And suite-level compounding, which nobody computes: **40 cases at `pass^k` 0.99
each gives 0.669 for the whole suite.**

### Per-category CI gates multiply your false-failure rate

Per-category thresholds are the right design — one global score lets a win in
one area hide a regression in another. It also does this:

```
   1 categories at alpha=0.05:  5.0% chance of at least one false failure
   3 categories at alpha=0.05: 14.3%
   6 categories at alpha=0.05: 26.5%
  12 categories at alpha=0.05: 46.0%
  20 categories at alpha=0.05: 64.2%
```

Twelve categories at 5% each means **roughly every other clean PR goes red**.
The team learns to re-run until it passes, and the gate is now decorative —
worse than no gate, because it still produces the badge. `gate.py` applies Holm,
and adds a third verdict: **`INCONCLUSIVE` blocks the merge exactly as `FAIL`
does**, because treating "we could not tell" as "fine" is how an eval suite
stops being a control.

### recall@k has a ceiling you did not choose

With `R` relevant documents and cutoff `k`, recall@k can never exceed `k/R`. A
system reporting recall@5 = 0.25 against 20 relevant documents retrieved every
document it had room for. It is perfect, and it reads as a 75% failure.

```
retrieval @5 over 1 queries
  recall@5    0.250   ceiling 0.250
  attainment    1.000   <- read THIS, not recall, when k < relevant count
  CEILING BINDING on 1/1 queries. Raw recall understates performance.
```

---

## What's in here

### `evalkit` — judge alignment, done properly

| Module | Contents |
|---|---|
| `metrics.py` | `Confusion`, precision/recall/specificity, `f_beta`, `mcc`, `wilson_interval` |
| `thresholds.py` | `candidate_thresholds`, `best_threshold`, `stratified_folds`, `tune_cv`, `feasible_region` |
| `calibration.py` | `brier`, `log_loss`, `reliability_bins`, `ece`, `brier_decomposition`, `PlattScaler`, `IsotonicCalibrator` |
| `compare.py` | `mcnemar`, `paired_bootstrap_delta`, `bootstrap_metric`, `pareto_frontier`, `cohens_kappa`, `cost_per_month` |
| `leakage.py` | `degenerate_baselines`, `ceiling_violation`, `grouping_gap`, out-of-fold item identity |
| `data.py` | the synthetic dataset and its five judges |
| `plot.py` | standalone SVG: score distributions, PR-vs-threshold, reliability diagrams |

### `evalinfra` — the infrastructure layer

Six modules covering the places standard practice names a measurement and stops
before the thing that tells you whether to believe it.

| Module | Supplies |
|---|---|
| `passk.py` | `pass@k` vs `pass^k`, certification, planning, suite compounding |
| `pairwise.py` | order-swap judging — plus the cost of the standard fix |
| `trajectory.py` | agent tool-path scoring with the single-reference assumption made measurable |
| `retrieval.py` | recall/nDCG ceilings, attainment, and the joint 2×2 |
| `assertions.py` | deterministic checks with taxonomy coverage accounting |
| `gate.py` | per-category CI gating, Holm, MDE, three verdicts |

### `benchaudit` — benchmark integrity

Near-duplicate detection, template multiplicity, and the design effect:
`D = 1 + (m̄−1)·ICC`, `n_eff = n/D`. At ICC 0.5 with 60 items per cluster, an
effective n of 30 needs 16 clusters.

---

## Two things the standard advice gets wrong

**Discarding inconsistent pairwise verdicts is a selection on outcome.** The
advice to run every A/B comparison twice with swapped order and count only
consistent verdicts is right about the disease and quiet about the side effect
— the pairs a judge flips on are the close ones, so what survives is an easier
population than you sampled:

```
pairs judged in both orders: 20
  consistent: 10  discarded: 50.0%
  A win rate, consistent only : 1.000   <- the standard figure
  A win rate, ties as ties    : 0.750   <- conservative
  VERDICT: NO SINGLE NUMBER IS QUOTABLE.
```

**An assertion suite's pass rate is not a quality metric.** It is a pass rate
over the failure modes the assertions can see, and a suite at 100% covering two
of four categories renders identically to one covering all four:

```
  raw pass rate   1.000
  taxonomy coverage 0.500 (2/4 categories)
    too_long                     NO ASSERTION
    hallucinated_citation        NO ASSERTION
  quality estimate: NONE — 2 of 4 failure categories have no assertion
  The pass rate above describes the categories that have assertions. It is
  not a statement about the system.
```

---

## The design rule, and it is the argument of the whole course

> **A function that cannot honestly answer returns `None` and reports the
> shortfall. It never relaxes a constraint to produce a number.**

`best_threshold` with unreachable floors returns `None`. `tune_cv` counts the
folds where no threshold qualified and prints `NONE` rather than a best-effort
F1. `design_effect` refuses on a single cluster, on singleton clusters, and on
constant values — and when it clamps a negative ICC to zero, it says so and
marks the figure an upper bound.

If your real eval tooling does not behave this way, it is answering questions
you did not ask.

---

## Errors this code caught in its own author

1. **"87 independent items in 100 rows"** on a fixture built from 5 templates.
   Word-trigram shingling was too brittle — one changed token destroys 3 of 9
   trigrams — and severity was being read from a single threshold. Fixed:
   `choose_representation` picks char-5 for short items and discloses the
   choice; grading reads the whole sweep. Now recovers exactly 5.
2. **Near-duplicate guessing outranked a declared grouping column.** The guess
   said "1 to 480," the donor column said 13. The declaration is authoritative;
   the guess is now a labelled cross-check.
3. **A ceiling-violation misdiagnosis.** Treating "a probe other than
   flag-everything fired" as leakage. The correct rule is "did any probe score
   above the trivial baseline."
4. **`ttl` matched inside "se*ttl*ement"**, counting a settlement requirement as
   expiry coverage. Whole-word matching for needles under 5 characters.
5. **A clean-build gate fixture asserted PASS and got INCONCLUSIVE.** MDE 5.9pp
   against a 2.0pp allowance at n=400. The tool was right; I fixed the test and
   kept the case as its own assertion.
6. **The test suites gave wrong answers under pytest, in both directions.** The
   function-style suites recorded failures without raising, so pytest showed a
   false green on broken code; the module-level suites ran their checks at
   import with nothing to collect, so pytest showed a false red ("no tests
   ran", exit 5) on passing code. Both are fixed — script and pytest runners now
   agree, verified by breaking the code on purpose under each.

## Layout

```
evalkit/      judge alignment, thresholds, calibration, comparison, leakage
evalinfra/    pass^k, pairwise, trajectory, retrieval, assertions, CI gating
benchaudit/   near-duplicates, template multiplicity, design effect
labs/         eight self-contained teaching labs
tests/        test_evalkit (134), test_evalinfra (232), test_leakage,
              test_benchaudit, test_design
```

## Licence

MIT.
