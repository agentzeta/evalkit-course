"""A synthetic eval dataset, deterministic and pedagogically rigged.

Real traces are better than synthetic ones and you should use your own the
moment you have them. This dataset exists so every lab runs identically on
every laptop, and so each teaching point has a judge that demonstrates it.

HOW IT IS BUILT, because the construction is itself a lesson.

Each trace gets a latent failure propensity pi, drawn from a two-component
mixture (most traces are nearly safe; a minority are genuinely risky). The
label is then drawn as Bernoulli(pi). So pi is the TRUE probability that the
trace contains a failure, and a judge reporting pi is perfectly calibrated by
construction -- not approximately, exactly.

That gives us a ground truth that most synthetic datasets lack, and it makes
`judge_oracle` the control case: a judge that cannot be improved by post-hoc
calibration, because there is nothing left to fix. Any calibration method that
claims to improve it is overfitting, and you can watch that happen.

The five judges:

  judge_oracle    reports pi exactly. Perfectly calibrated, strong
                  discrimination. The control -- calibration should NOT help,
                  and may hurt.
  judge_strong    good discrimination, systematically mis-scaled -> Platt helps
  judge_weak      little discrimination. Low resolution. Nothing helps; the
                  honest conclusion is to rewrite the prompt, not to calibrate.
  judge_shifted   strong discrimination, but every score falls below 0.5, so
                  the 0.50 default catches NOTHING. The whole threshold
                  argument lives in this one column.
  judge_recall    flags freely: high recall, poor precision.

Failure modes are unequal in frequency AND in how visible they are to a judge,
so a single broad judge underperforms per-mode judges -- the lesson of Module 2,
which cannot be taught on a single-mode dataset. The rarest mode appears about
twice in 200 traces, so students meet the "you cannot measure this" problem in
the data rather than in a slide.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

FAILURE_MODES = [
    "booking_failed",
    "missed_handoff",
    "wrong_unit_quoted",
    "hallucinated_amenity",
    "pii_in_response",
    "repeated_loop",
]

# Relative frequency among failing traces. Long-tailed on purpose.
_MODE_WEIGHTS = [0.30, 0.24, 0.18, 0.14, 0.10, 0.04]

# How visible each mode is to an automated judge. 1.0 = as visible as the
# latent propensity implies; below 1.0 the judge is pulled toward calling it a
# pass; above 1.0 it stands out. pii_in_response is near-pattern-matchable;
# missed_handoff needs conversational judgement and is the hardest.
_MODE_VISIBILITY = {
    "booking_failed": 1.10,
    "missed_handoff": 0.45,
    "wrong_unit_quoted": 0.75,
    "hallucinated_amenity": 0.70,
    "pii_in_response": 1.55,
    "repeated_loop": 0.90,
}

# Logit penalty applied at zero visibility. Scales the effect above.
_VISIBILITY_SCALE = 2.6

_SNIPPETS = {
    "booking_failed": [
        "User asked to book a tour for Saturday 2pm. Agent called "
        "schedule_tour(unit=12B, slot=sat_14) -> 409 CONFLICT. Agent replied "
        "'You're all set for Saturday at 2!' and ended the conversation.",
        "Agent attempted book_tour twice, both returned SLOT_TAKEN, then said "
        "'I've reserved that for you' without retrying or offering alternatives.",
    ],
    "missed_handoff": [
        "User: 'I want to dispute the deposit withholding, this is illegal.' "
        "Agent continued with policy explanation for 6 turns. No escalation to "
        "a human leasing agent. User left the conversation.",
        "User mentioned a service animal and ADA accommodation. Agent answered "
        "from the pet policy FAQ instead of routing to the compliance queue.",
    ],
    "wrong_unit_quoted": [
        "User asked about the 2-bed on the 4th floor. Agent quoted $2,150, "
        "which is the rate for unit 2C (1-bed). Correct rate for 4B is $2,850.",
        "Agent quoted the promotional rate that expired last month and told the "
        "user it was 'current pricing, locked for 30 days'.",
    ],
    "hallucinated_amenity": [
        "Agent told the user the building has a rooftop pool and co-working "
        "lounge. Property record lists neither. User asked to apply on that basis.",
        "Agent confirmed in-unit laundry for a unit where the record says "
        "'shared laundry, floor 1'.",
    ],
    "pii_in_response": [
        "Agent echoed the previous applicant's full name and the last four "
        "digits of their SSN while explaining the application status lookup.",
        "Agent pasted the raw CRM record, including another tenant's phone "
        "number and move-out date, into the chat.",
    ],
    "repeated_loop": [
        "Agent asked 'What's your target move-in date?' five times across nine "
        "turns despite the user answering 'March 1' each time.",
        "Agent cycled between availability and pricing tools 11 times without "
        "producing a response, then timed out.",
    ],
}

_PASS_SNIPPETS = [
    "User asked about pet policy. Agent quoted the correct policy (cats and "
    "dogs under 40lb, $300 deposit) and offered to schedule a tour.",
    "User asked for 1-beds under $2,000. Agent returned three matching units "
    "with correct rates and availability, then booked a tour for Thursday 5pm.",
    "User asked whether parking is included. Agent said no, quoted $125/month "
    "for the garage, and confirmed one space was available.",
    "User requested an application link. Agent sent the correct link and "
    "explained the $50 fee and required documents.",
    "User asked about the commute to downtown. Agent declined to estimate "
    "traffic and offered the building's address instead.",
    "User asked to reschedule Friday's tour to Monday. Agent cancelled and "
    "rebooked correctly, then confirmed both changes.",
]

EPS = 1e-6


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


@dataclass
class Trace:
    trace_id: str
    text: str
    label: int  # 1 = failure
    failure_mode: str | None
    propensity: float = 0.0  # the latent pi; a judge reporting it is calibrated
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class Dataset:
    traces: list[Trace]
    judges: list[str]

    def __len__(self) -> int:
        return len(self.traces)

    @property
    def labels(self) -> list[int]:
        return [t.label for t in self.traces]

    def scores(self, judge: str) -> list[float]:
        if judge not in self.judges:
            raise KeyError(f"unknown judge {judge!r}; have {self.judges}")
        return [t.scores[judge] for t in self.traces]

    @property
    def propensities(self) -> list[float]:
        """The latent truth. You never have this in real life -- that is the
        point of having it here exactly once."""
        return [t.propensity for t in self.traces]

    @property
    def base_rate(self) -> float:
        return sum(self.labels) / len(self.labels) if self.traces else 0.0

    def mode_labels(self, mode: str) -> list[int]:
        """Binary labels for ONE failure mode; everything else is a pass.

        This is what a per-mode judge should be measured against, and building
        it is the whole point of narrowing an evaluator's scope.
        """
        if mode not in FAILURE_MODES:
            raise KeyError(f"unknown mode {mode!r}")
        return [1 if t.failure_mode == mode else 0 for t in self.traces]

    def mode_counts(self) -> dict[str, int]:
        counts = {m: 0 for m in FAILURE_MODES}
        for t in self.traces:
            if t.failure_mode:
                counts[t.failure_mode] += 1
        return counts

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["trace_id", "label", "failure_mode", "propensity",
                 *self.judges, "text"]
            )
            for t in self.traces:
                w.writerow(
                    [
                        t.trace_id,
                        t.label,
                        t.failure_mode or "",
                        f"{t.propensity:.6f}",
                        *[f"{t.scores[j]:.6f}" for j in self.judges],
                        t.text,
                    ]
                )
        return path


# Judge specs: (gain on the latent logit, noise sd, bias, lo, hi).
# The score is lo + (hi-lo) * sigmoid(gain * z_eff + bias + noise), where z_eff
# is the latent logit adjusted for how visible the failure mode is.
#
# judge_oracle is handled separately: it reports pi untouched.
_JUDGE_SPECS = {
    "judge_strong": (0.95, 0.80, 0.30, 0.02, 0.98),
    "judge_weak": (0.30, 1.30, 0.10, 0.05, 0.95),
    "judge_shifted": (1.05, 0.55, 0.60, 0.05, 0.45),
    "judge_recall": (0.85, 0.75, 1.70, 0.05, 0.97),
}

JUDGES = ["judge_oracle", *_JUDGE_SPECS]


def make_dataset(
    n: int = 200, base_rate: float = 0.25, seed: int = 20261061
) -> Dataset:
    """Build the labelled dataset. Same seed, same bytes, every time.

    `base_rate` steers the mixture weight toward the requested failure rate;
    the realised rate varies by a point or two because labels are drawn, not
    assigned. That is deliberate: real base rates are estimates too.

    n=200 rather than the 100 you often see: at a 25% failure rate a five-fold
    split of 100 traces leaves five failures per test fold, and recall then
    moves in steps of 0.20. Students should see stable numbers first and meet
    the small-sample problem on purpose, via `subsample` in Lab 7.
    """
    rng = random.Random(seed)

    # Two-component mixture for the latent propensity.
    #   safe  ~ Beta(1.0, 30)   mean 0.0323
    #   risky ~ Beta(9.0, 1.6)  mean 0.8491
    # Mixture weight w on the risky component solves
    #   w*0.8491 + (1-w)*0.0323 = base_rate.
    #
    # The components are deliberately far apart. Because labels are DRAWN from
    # pi, the gap sets the irreducible label noise -- and therefore the ceiling
    # on any judge's F1, including the oracle's. Lab 7 uses that ceiling.
    mean_safe, mean_risky = 1.0 / 31.0, 9.0 / 10.6
    w_risky = (base_rate - mean_safe) / (mean_risky - mean_safe)
    w_risky = min(0.95, max(0.01, w_risky))

    traces: list[Trace] = []
    for i in range(n):
        if rng.random() < w_risky:
            pi = rng.betavariate(9.0, 1.6)
        else:
            pi = rng.betavariate(1.0, 30.0)

        is_fail = rng.random() < pi

        if is_fail:
            mode = rng.choices(FAILURE_MODES, weights=_MODE_WEIGHTS, k=1)[0]
            text = rng.choice(_SNIPPETS[mode])
            visibility = _MODE_VISIBILITY[mode]
        else:
            mode = None
            text = rng.choice(_PASS_SNIPPETS)
            visibility = 1.0

        t = Trace(
            trace_id=f"trace_{i:04d}",
            text=text,
            label=1 if is_fail else 0,
            failure_mode=mode,
            propensity=pi,
        )

        # The oracle reports the latent probability, untouched. It is therefore
        # calibrated by construction and is the control for Lab 4.
        t.scores["judge_oracle"] = pi

        z = _logit(pi)
        # A hard-to-see failure mode drags the judge's view toward "pass".
        z_eff = z - _VISIBILITY_SCALE * (1.0 - visibility) if is_fail else z

        for name, (gain, noise, bias, lo, hi) in _JUDGE_SPECS.items():
            raw = _sigmoid(gain * z_eff + bias + rng.gauss(0.0, noise))
            t.scores[name] = lo + (hi - lo) * raw

        traces.append(t)

    return Dataset(traces=traces, judges=list(JUDGES))


def subsample(ds: Dataset, n: int, seed: int = 0) -> Dataset:
    """A smaller dataset drawn from the same one, for sample-size experiments.

    Run the same tuning on n=200 then n=50 and watch the chosen threshold move.
    That demonstration retires more bad eval practice than any slide.
    """
    rng = random.Random(seed)
    picked = rng.sample(ds.traces, min(n, len(ds.traces)))
    return Dataset(traces=picked, judges=list(ds.judges))


# Cents per 100 eval runs. Illustrative figures for the cost lab -- replace
# them with your provider's prices and your own measured token counts. Never
# quote vendor pricing from a course handout.
EXAMPLE_COSTS = {
    "judge_oracle": 4.80,
    "judge_strong": 3.72,
    "judge_weak": 0.31,
    "judge_shifted": 0.82,
    "judge_recall": 0.55,
}
