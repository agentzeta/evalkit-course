"""pass@k and pass^k — the arithmetic of reliability under resampling.

The distinction matters more than it looks.

    pass@k  = succeeded AT LEAST ONCE in k runs   (capability)
    pass^k  = succeeded in ALL k runs             (reliability)

Production lives in pass^k. At a per-run success rate of 0.80, pass^5 is
0.328 — the system you measured at "80%" fails a five-step chain two times
in three.

The part almost nobody accounts for: **pass^k is far harder to certify than
pass@1, because it amplifies the error in your estimate of p.** The
derivative of p^k is k*p^(k-1), so at p=0.80 and k=5 a one-point error in p
becomes a two-point error in pass^5. The consequence is concrete and it is
the reason this module exists:

    You measure 80/100 and report pass^5 = 0.328.
    The honest lower bound is 0.183.
    You cannot certify pass^5 >= 0.30 from n=100. Not even close.

Functions that cannot honestly answer return None and report the shortfall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "pass_at_k",
    "pass_hat_k",
    "pass_hat_k_plugin",
    "required_per_run_rate",
    "PassKReport",
    "assess",
    "n_for_pass_hat_k",
    "suite_all_pass",
]


# --------------------------------------------------------------------------
# point estimators
# --------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: P(at least one success among k drawn without replacement).

    n total runs, c of them successful. This is the standard estimator,
    1 - C(n-c, k) / C(n, k).
    """
    _check(n, c, k)
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k: P(all k succeed among k drawn without replacement).

    C(c, k) / C(n, k). The exact mirror of pass@k, and the one that matters
    for a system whose k steps must all work.
    """
    _check(n, c, k)
    return _comb(c, k) / _comb(n, k)


def pass_hat_k_plugin(p: float, k: int) -> float:
    """The naive plug-in p**k.

    Kept separate and named "plugin" on purpose. It is what everyone writes
    down, it is biased upward when p is itself an estimate, and `assess`
    reports it next to the unbiased figure so the gap is visible.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return p ** k


def required_per_run_rate(target: float, k: int) -> float:
    """The per-run success rate needed to reach `target` pass^k.

    Inverts p**k = target. This is the number to put in front of anyone who
    says "80% is fine": for pass^5 >= 0.95 you need p >= 0.98977.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return target ** (1.0 / k)


# --------------------------------------------------------------------------
# intervals — the part that changes conclusions
# --------------------------------------------------------------------------

def _wilson(k_succ: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Correct at small n and at the
    boundaries, where the normal approximation is not."""
    if n == 0:
        return (0.0, 1.0)
    p = k_succ / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class PassKReport:
    n: int
    c: int
    k: int
    p_hat: float
    p_lo: float
    p_hi: float
    pass_at_k: float
    pass_hat_k: float
    pass_hat_k_plugin: float
    pass_hat_k_lo: float
    pass_hat_k_hi: float
    amplification: float
    target: float | None
    certified: bool | None

    def summary(self) -> str:
        lines = [
            f"n={self.n} runs, c={self.c} successes, k={self.k}",
            f"  per-run p          {self.p_hat:.3f}  "
            f"[{self.p_lo:.3f}, {self.p_hi:.3f}]",
            f"  pass@{self.k} (capability) {self.pass_at_k:.3f}",
            f"  pass^{self.k} (reliability) {self.pass_hat_k:.3f}  "
            f"[{self.pass_hat_k_lo:.3f}, {self.pass_hat_k_hi:.3f}]",
            f"  plug-in p^{self.k}        {self.pass_hat_k_plugin:.3f}"
            f"   (biased upward when p is estimated)",
            f"  error amplification  {self.amplification:.2f}x"
            f"   (one point of error in p becomes "
            f"{self.amplification:.2f} points in pass^{self.k})",
        ]
        if self.target is not None:
            verdict = "CERTIFIED" if self.certified else "NOT CERTIFIED"
            lines.append(
                f"  target pass^{self.k} >= {self.target:.3f}: {verdict}"
            )
            if not self.certified:
                need = required_per_run_rate(self.target, self.k)
                lines.append(
                    f"    point estimate {self.pass_hat_k:.3f} is above target, "
                    f"but the lower bound {self.pass_hat_k_lo:.3f} is not."
                    if self.pass_hat_k >= self.target
                    else f"    point estimate {self.pass_hat_k:.3f} "
                         f"is below target."
                )
                lines.append(
                    f"    certification needs the lower bound on p "
                    f"above {need:.5f}; it is {self.p_lo:.5f}."
                )
        return "\n".join(lines)


def assess(
    n: int,
    c: int,
    k: int,
    target: float | None = None,
    z: float = 1.96,
) -> PassKReport:
    """Full accounting for a pass^k claim from n runs with c successes.

    `target`, if given, is certified only when the LOWER confidence bound on
    pass^k clears it. A point estimate above target is not certification and
    the report says so explicitly.
    """
    _check(n, c, k)
    p_hat = c / n
    p_lo, p_hi = _wilson(c, n, z)
    # pass^k is monotone increasing in p, so the interval maps through directly
    phk_lo = p_lo ** k
    phk_hi = p_hi ** k
    amp = k * (p_hat ** (k - 1)) if p_hat > 0 else float(k)
    certified = None if target is None else bool(phk_lo >= target)
    return PassKReport(
        n=n,
        c=c,
        k=k,
        p_hat=p_hat,
        p_lo=p_lo,
        p_hi=p_hi,
        pass_at_k=pass_at_k(n, c, k),
        pass_hat_k=pass_hat_k(n, c, k),
        pass_hat_k_plugin=pass_hat_k_plugin(p_hat, k),
        pass_hat_k_lo=phk_lo,
        pass_hat_k_hi=phk_hi,
        amplification=amp,
        target=target,
        certified=certified,
    )


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def n_for_pass_hat_k(
    target: float,
    k: int,
    assumed_p: float,
    z: float = 1.96,
    max_n: int = 2_000_000,
) -> int | None:
    """Runs needed before pass^k >= target could be certified at `assumed_p`.

    Returns None if `assumed_p` is at or below the required rate — no amount
    of data certifies a target the system does not meet. That refusal is the
    point: the function will not hand back a number that implies more data
    would fix a capability shortfall.
    """
    need = required_per_run_rate(target, k)
    if assumed_p <= need:
        return None
    lo, hi = 1, 1024
    while hi <= max_n:
        c = int(round(assumed_p * hi))
        if _wilson(c, hi, z)[0] >= need:
            break
        lo, hi = hi, hi * 2
    else:
        return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        c = int(round(assumed_p * mid))
        if _wilson(c, mid, z)[0] >= need:
            hi = mid
        else:
            lo = mid
    return hi


def suite_all_pass(per_case: Sequence[float]) -> float:
    """P(every case in a suite passes), given each case's pass^k.

    Compounding is the thing people forget. Forty cases at 0.99 each gives
    0.669 for the suite. Independence is assumed and is usually optimistic,
    so read this as an upper bound.
    """
    out = 1.0
    for p in per_case:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probabilities must be in [0, 1], got {p}")
        out *= p
    return out


# --------------------------------------------------------------------------

def _check(n: int, c: int, k: int) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(
            f"k must be in [1, n]; got k={k}, n={n}. "
            f"Estimating pass^{k} needs at least {k} runs per case."
        )
