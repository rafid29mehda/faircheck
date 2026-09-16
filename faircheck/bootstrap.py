"""Bootstrap confidence intervals for the disaggregated metrics.

**The fast path.** Every rate in :mod:`faircheck.metrics` is a function of the four
confusion-matrix cell counts and nothing else. Resampling ``n_g`` rows with replacement
within group *g* therefore induces exactly a ``Multinomial(n_g, cells_g / n_g)`` draw over
those four cells -- the row identities are irrelevant once a row's cell is known. So the
whole bootstrap is one multinomial draw per group: cost ``O(n_boot x 4 x n_groups)``,
independent of how many rows the user uploaded, with no ``(n_boot, n_rows)`` index matrix.

This is an exact restatement of the row-level bootstrap, not an approximation, and
``tests/test_bootstrap.py`` checks it against a naive row-resampling implementation rather
than taking it on trust.

**The trap this module is careful about.** A gap defined as ``max_a r_a - min_a r_a`` is
non-negative by construction, so its confidence interval almost never contains zero even
when every group shares the same true rate -- sampling noise alone pushes the maximum above
the minimum. "The CI excludes zero" is therefore *not* evidence of a disparity for these
statistics, and a tool that says otherwise will report a gap in every dataset it ever sees.
For evidence claims this module provides **signed contrasts against a reference group**,
which can legitimately straddle zero. See docs/DECISIONS.md D11.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from faircheck import metrics
from faircheck.types import N_CELLS, FloatArray, GroupCounts, IntArray

DEFAULT_N_BOOT: Final = 1000

#: Percentile bootstrap at the 95% level, with the median carried through so the report can
#: show how far the resampling median sits from the point estimate.
DEFAULT_QUANTILES: Final = (0.025, 0.5, 0.975)

STRATIFIED: Final = "stratified"
FULL: Final = "full"
SCHEMES: Final = (STRATIFIED, FULL)


def _pvals(counts: FloatArray) -> FloatArray:
    """Normalise counts to multinomial probabilities.

    ``Generator.multinomial`` rejects ``pvals`` whose leading entries sum past 1 by even one
    ulp, which plain division can produce, so the last entry absorbs the rounding error.
    """
    probabilities = counts / counts.sum()
    probabilities[-1] = max(0.0, 1.0 - float(probabilities[:-1].sum()))
    return probabilities


def bootstrap_counts(
    counts: GroupCounts,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    scheme: str = STRATIFIED,
) -> IntArray:
    """Draw ``n_boot`` resampled confusion tables, shaped ``(n_boot, n_groups, 4)``.

    ``scheme="stratified"`` holds each group's size at its observed value, which is the right
    conditioning for per-group rate intervals: we are asking how precisely this group's rate
    is estimated, not how many members of the group might have turned up.

    ``scheme="full"`` resamples the dataset as a whole, so group sizes vary between
    resamples. Both are exact; ``full`` matches what Fairlearn's ``MetricFrame`` bootstrap
    does and gives slightly wider intervals for small groups.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")

    rng = np.random.default_rng(seed)
    cells = counts.cells.astype(np.float64)

    if scheme == FULL:
        flat = cells.ravel()
        draws = rng.multinomial(int(flat.sum()), _pvals(flat), size=n_boot)
        return draws.reshape(n_boot, counts.n_groups, N_CELLS).astype(np.int64)

    resampled = np.empty((n_boot, counts.n_groups, N_CELLS), dtype=np.int64)
    for index in range(counts.n_groups):
        group_size = int(cells[index].sum())
        resampled[:, index, :] = rng.multinomial(group_size, _pvals(cells[index]), size=n_boot)
    return resampled


def _quantiles(samples: FloatArray, quantiles: Sequence[float]) -> FloatArray:
    """Percentiles over the resample axis, shaped ``(len(quantiles), *samples.shape[1:])``.

    Resamples in which a metric is undefined are skipped rather than poisoning the whole
    interval: a group with three positives will lose a handful of resamples to zero
    positives, and discarding the interval entirely would be less informative than
    reporting it next to the undefined fraction. Where *every* resample is undefined the
    interval is NaN, because there is genuinely nothing to report.
    """
    all_undefined = np.all(~np.isfinite(samples), axis=0)
    # Placeholder values keep nanquantile off all-NaN slices (which warn); the results at
    # those positions are overwritten with NaN immediately below.
    safe = np.where(all_undefined[None, ...], 0.0, samples)
    computed = np.nanquantile(safe, quantiles, axis=0)
    return np.asarray(np.where(all_undefined[None, ...], np.nan, computed), dtype=np.float64)


def _undefined_fraction(samples: FloatArray) -> FloatArray:
    return np.asarray(1.0 - np.isfinite(samples).mean(axis=0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Point estimates and percentile intervals for every rate, gap and contrast.

    Arrays are indexed ``[quantile, group]`` for per-group quantities and ``[quantile]`` for
    gaps, with the quantile axis ordered as :attr:`quantiles`.
    """

    quantiles: tuple[float, ...]
    group_labels: tuple[str, ...]
    reference_index: int
    n_boot: int
    seed: int
    scheme: str

    rates: dict[str, FloatArray]
    rate_ci: dict[str, FloatArray]
    rate_undefined_fraction: dict[str, FloatArray]

    gaps: dict[str, FloatArray]
    gap_ci: dict[str, FloatArray]

    contrasts: dict[str, FloatArray]
    contrast_ci: dict[str, FloatArray]

    @property
    def reference_label(self) -> str:
        return self.group_labels[self.reference_index]

    def _bounds(self, array: FloatArray) -> tuple[float, float]:
        return float(array[0]), float(array[-1])

    def rate_interval(self, key: str, group_index: int) -> tuple[float, float, float]:
        """``(low, point, high)`` for one group's rate."""
        low, high = self._bounds(self.rate_ci[key][:, group_index])
        return low, float(self.rates[key][group_index]), high

    def gap_interval(self, key: str) -> tuple[float, float, float]:
        low, high = self._bounds(self.gap_ci[key])
        return low, float(self.gaps[key]), high

    def contrast_interval(self, key: str, group_index: int) -> tuple[float, float, float]:
        """``(low, point, high)`` for this group's rate minus the reference group's."""
        low, high = self._bounds(self.contrast_ci[key][:, group_index])
        return low, float(self.contrasts[key][group_index]), high

    def has_clear_contrast(self, key: str, group_index: int) -> bool | None:
        """Whether this group's signed contrast against the reference excludes zero.

        ``None`` when the interval could not be computed at all. A ``False`` here means
        "no clear evidence of a difference", which is emphatically not "no difference" --
        the report's wording has to preserve that distinction.
        """
        low, _, high = self.contrast_interval(key, group_index)
        if not (np.isfinite(low) and np.isfinite(high)):
            return None
        return bool(low > 0.0 or high < 0.0)


def bootstrap_audit(
    counts: GroupCounts,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    scheme: str = STRATIFIED,
    reference_index: int | None = None,
) -> BootstrapResult:
    """Compute every rate, gap and reference contrast with percentile intervals.

    ``reference_index`` defaults to the largest group, which keeps the contrasts as precise
    as the data allows and avoids picking the extreme group *after* looking at the results
    (doing that biases the interval, because the winner of a noisy maximum is optimistic).
    """
    quantile_tuple = tuple(float(q) for q in quantiles)
    if not quantile_tuple:
        raise ValueError("At least one quantile is required.")
    if any(not 0.0 < q < 1.0 for q in quantile_tuple):
        raise ValueError(f"Quantiles must lie strictly between 0 and 1, got {quantile_tuple}")
    if list(quantile_tuple) != sorted(quantile_tuple):
        raise ValueError(f"Quantiles must be ascending, got {quantile_tuple}")

    reference = int(np.argmax(counts.sizes)) if reference_index is None else reference_index
    if not 0 <= reference < counts.n_groups:
        raise ValueError(
            f"reference_index {reference} is outside the {counts.n_groups} observed groups"
        )

    observed_rates = metrics.per_group_rates(counts.cells)
    observed_gaps = metrics.compute_gaps(observed_rates)

    resampled = bootstrap_counts(counts, n_boot=n_boot, seed=seed, scheme=scheme)
    sampled_rates = metrics.per_group_rates(resampled)
    sampled_gaps = metrics.compute_gaps(sampled_rates)

    return BootstrapResult(
        quantiles=quantile_tuple,
        group_labels=counts.labels(),
        reference_index=reference,
        n_boot=n_boot,
        seed=seed,
        scheme=scheme,
        rates=dict(observed_rates),
        rate_ci={key: _quantiles(value, quantile_tuple) for key, value in sampled_rates.items()},
        rate_undefined_fraction={
            key: _undefined_fraction(value) for key, value in sampled_rates.items()
        },
        gaps={key: np.asarray(value) for key, value in observed_gaps.items()},
        gap_ci={key: _quantiles(value, quantile_tuple) for key, value in sampled_gaps.items()},
        contrasts=_contrasts(observed_rates, reference),
        contrast_ci={
            key: _quantiles(value, quantile_tuple)
            for key, value in _contrasts(sampled_rates, reference).items()
        },
    )


def _contrasts(rates: Mapping[str, FloatArray], reference: int) -> dict[str, FloatArray]:
    """Signed difference between each group's rate and the reference group's.

    Unlike ``max - min``, this is symmetric about zero under the null, so its interval can
    contain zero and therefore supports an honest "no clear evidence" verdict.
    """
    return {key: value - value[..., reference, None] for key, value in rates.items()}
