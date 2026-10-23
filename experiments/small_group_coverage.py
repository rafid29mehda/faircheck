"""Coverage of 95% intervals for small groups: resampling bootstrap vs Dirichlet posterior.

Evidence for D15. Run from the repo root:

    .venv/bin/python -m experiments.small_group_coverage

Two tables, both deterministic (fixed seeds):

1. **Fixed truth.** A group of ``n`` rows with base rate 0.3 and a known TPR; how often does
   each interval contain the true TPR? This is the neutral comparison.
2. **By smallest observed cell.** Random truths drawn from a Dirichlet(0.7) over the four
   cells, bucketed by the smallest observed cell, over all four verdict rates. Because the
   truths come from a Dirichlet, this favours the posterior by construction; read it for
   *where* the bootstrap breaks (a zero cell), not for the size of the gap.
"""

from __future__ import annotations

import numpy as np

from faircheck.bootstrap import POSTERIOR_PRIOR, bootstrap_counts, posterior_counts
from faircheck.metrics import per_group_rates
from faircheck.types import FloatArray, GroupCounts

SEED = 0
RATE_KEYS = ("selection_rate", "tpr", "fpr", "precision")
BUCKETS = ((0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 10**9))


def _single_group(cells: np.ndarray) -> GroupCounts:
    return GroupCounts(
        group_names=("g",), groups=(("A",),), cells=cells.reshape(1, 4).astype(np.int64)
    )


def _interval(values: FloatArray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _draws(counts: GroupCounts, n_boot: int, seed: int) -> dict[str, FloatArray]:
    """Resampled rates per method for a one-group table, shaped ``(n_boot,)`` each."""
    boot = bootstrap_counts(counts, n_boot=n_boot, seed=seed).astype(np.float64)
    post = posterior_counts(counts, n_boot=n_boot, seed=seed)
    return {
        "bootstrap": per_group_rates(boot)["tpr"][:, 0],
        "dirichlet": per_group_rates(post)["tpr"][:, 0],
    }


def fixed_truth(
    n_values: tuple[int, ...], tprs: tuple[float, ...], *, trials: int, n_boot: int
) -> list[tuple[int, float, float, float]]:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in n_values:
        for tpr in tprs:
            cell_p = np.array([0.3 * tpr, 0.7 * 0.2, 0.3 * (1 - tpr), 0.7 * 0.8])
            hits = {"bootstrap": 0, "dirichlet": 0}
            used = 0
            for trial in range(trials):
                cells = rng.multinomial(n, cell_p)
                if cells[0] + cells[2] == 0:
                    continue  # TPR undefined: reported as n/a, not a coverage question
                used += 1
                for method, draws in _draws(_single_group(cells), n_boot, trial).items():
                    interval = _interval(draws)
                    hits[method] += interval is not None and interval[0] <= tpr <= interval[1]
            rows.append((n, tpr, hits["bootstrap"] / used, hits["dirichlet"] / used))
    return rows


def by_smallest_cell(*, trials: int, n_boot: int) -> list[tuple[str, int, float, float]]:
    rng = np.random.default_rng(SEED + 1)
    tally = {(m, b): [0, 0] for m in ("bootstrap", "dirichlet") for b in range(len(BUCKETS))}
    for trial in range(trials):
        n = int(np.exp(rng.uniform(np.log(20), np.log(3000))))
        truth = rng.dirichlet(np.full(4, 0.7))
        true_rates = per_group_rates(truth)
        cells = rng.multinomial(n, truth)
        bucket = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= cells.min() <= hi)
        counts = _single_group(cells)
        observed = per_group_rates(counts.cells)
        boot = per_group_rates(bootstrap_counts(counts, n_boot=n_boot, seed=trial).astype(float))
        post = per_group_rates(posterior_counts(counts, n_boot=n_boot, seed=trial))
        for method, rates in (("bootstrap", boot), ("dirichlet", post)):
            for key in RATE_KEYS:
                if not (np.isfinite(observed[key][0]) and np.isfinite(true_rates[key])):
                    continue
                interval = _interval(rates[key][:, 0])
                if interval is None:
                    continue
                cell = tally[(method, bucket)]
                cell[0] += interval[0] <= float(true_rates[key]) <= interval[1]
                cell[1] += 1
    return [
        (
            f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
            tally[("bootstrap", b)][1],
            tally[("bootstrap", b)][0] / max(tally[("bootstrap", b)][1], 1),
            tally[("dirichlet", b)][0] / max(tally[("dirichlet", b)][1], 1),
        )
        for b, (lo, hi) in enumerate(BUCKETS)
    ]


def main() -> None:
    print(f"Dirichlet prior per cell: {POSTERIOR_PRIOR}\n")
    print("Fixed truth: coverage of the 95% interval for TPR (base rate 0.3)")
    print(f"{'n':>5} {'true TPR':>9} {'bootstrap':>10} {'dirichlet':>10}")
    for n, tpr, boot_cov, dir_cov in fixed_truth(
        (20, 40, 100, 400), (0.5, 0.8, 0.95), trials=1_500, n_boot=600
    ):
        print(f"{n:>5} {tpr:>9.2f} {boot_cov:>10.3f} {dir_cov:>10.3f}")
    print("\nBy smallest observed cell (4 verdict rates; truths ~ Dirichlet(0.7))")
    print(f"{'min cell':>9} {'cases':>7} {'bootstrap':>10} {'dirichlet':>10}")
    for label, cases, boot_cov, dir_cov in by_smallest_cell(trials=12_000, n_boot=300):
        print(f"{label:>9} {cases:>7} {boot_cov:>10.3f} {dir_cov:>10.3f}")


if __name__ == "__main__":
    main()
