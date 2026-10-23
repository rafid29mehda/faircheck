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
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from faircheck import metrics
from faircheck.types import N_CELLS, FloatArray, GroupCounts, IntArray

DEFAULT_N_BOOT: Final = 1000

#: Percentile bootstrap at the 95% level, with the median carried through so the report can
#: show how far the resampling median sits from the point estimate.
DEFAULT_QUANTILES: Final = (0.025, 0.5, 0.975)

#: The rates whose group-versus-reference contrasts drive every verdict and summary
#: sentence. These are the inputs of the five gaps, so they are the multiplicity family.
VERDICT_RATE_KEYS: Final = ("selection_rate", "tpr", "fpr", "precision")

MAX_T: Final = "max_t"
NO_ADJUSTMENT: Final = "none"
ADJUSTMENTS: Final = (MAX_T, NO_ADJUSTMENT)
FAMILY_LEVEL: Final = 0.95

#: How a group's resampled cell tables are produced (D15). ``adaptive`` picks per group:
#: the resampling bootstrap (D4) where every cell is well populated, a Dirichlet posterior
#: over the four cells where one is not. The bootstrap can only reproduce cells it has seen,
#: so with a zero or tiny cell it is degenerate or far too narrow.
BOOTSTRAP: Final = "bootstrap"
DIRICHLET: Final = "dirichlet"
ADAPTIVE: Final = "adaptive"
INTERVAL_METHODS: Final = (ADAPTIVE, BOOTSTRAP, DIRICHLET)

#: A group whose smallest cell is below this uses the posterior. Chosen from the coverage
#: study in D15: the bootstrap is adequate once every observed cell is at least 2, and 5
#: leaves a margin.
MIN_CELL_FOR_BOOTSTRAP: Final = 5

#: Pseudo-count added to each cell. 0.5 is the Jeffreys prior; aggregating cells, every
#: rate that is a ratio of two cell sums then has a Beta posterior with that prior.
POSTERIOR_PRIOR: Final = 0.5

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


def choose_methods(counts: GroupCounts, interval_method: str) -> tuple[str, ...]:
    """The resampler for each group: forced for all, or ``adaptive`` on the smallest cell."""
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(
            f"interval_method must be one of {INTERVAL_METHODS}, got {interval_method!r}"
        )
    if interval_method != ADAPTIVE:
        return (interval_method,) * counts.n_groups
    return tuple(
        DIRICHLET if int(counts.cells[index].min()) < MIN_CELL_FOR_BOOTSTRAP else BOOTSTRAP
        for index in range(counts.n_groups)
    )


def posterior_counts(
    counts: GroupCounts,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    prior: float = POSTERIOR_PRIOR,
) -> FloatArray:
    """Posterior draws of each group's cell proportions, shaped ``(n_boot, n_groups, 4)``.

    ``Dirichlet(cells + prior)`` per group. Every rate is a function of the proportions
    alone (the metric functions are scale-free), so these draws go through the same
    formulas as resampled counts. Uses its own seed stream so adding a small group never
    changes the bootstrap draws of the others.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    rng = np.random.default_rng([seed, 1])
    draws = np.empty((n_boot, counts.n_groups, N_CELLS), dtype=np.float64)
    for index in range(counts.n_groups):
        draws[:, index, :] = rng.dirichlet(counts.cells[index] + prior, size=n_boot)
    return draws


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

    #: Simultaneous (max-t) bands for the verdict contrasts, ``[low/high, group]`` per rate in
    #: :data:`VERDICT_RATE_KEYS`. Empty when ``adjustment == "none"``. See D14.
    adjustment: str = NO_ADJUSTMENT
    adjusted_contrast_ci: dict[str, FloatArray] = field(default_factory=dict)
    family_size: int = 0
    critical_value: float = float("nan")

    #: Per group, which resampler produced its intervals (D15).
    interval_methods: tuple[str, ...] = ()

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

    def has_clear_unadjusted_contrast(self, key: str, group_index: int) -> bool | None:
        """The marginal 95% rule for one contrast, ignoring the other comparisons."""
        low, _, high = self.contrast_interval(key, group_index)
        if not (np.isfinite(low) and np.isfinite(high)):
            return None
        return bool(low > 0.0 or high < 0.0)

    def has_clear_contrast(self, key: str, group_index: int) -> bool | None:
        """Whether this group's signed contrast against the reference excludes zero.

        For the headline rates this uses the simultaneous band (D14), so the verdict holds
        across every contrast in the family at once; ``adjust="none"`` and rates outside the
        family use the marginal interval. ``None`` when the interval could not be computed.
        A ``False`` means "no clear evidence of a difference", which is emphatically not "no
        difference" -- the report's wording has to preserve that distinction.
        """
        if self.adjustment == MAX_T and key in self.adjusted_contrast_ci:
            low, high = self._bounds(self.adjusted_contrast_ci[key][:, group_index])
            if not (np.isfinite(low) and np.isfinite(high)):
                return None
            return bool(low > 0.0 or high < 0.0)
        return self.has_clear_unadjusted_contrast(key, group_index)


def bootstrap_audit(
    counts: GroupCounts,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    scheme: str = STRATIFIED,
    reference_index: int | None = None,
    adjust: str = MAX_T,
    interval_method: str = ADAPTIVE,
) -> BootstrapResult:
    """Compute every rate, gap and reference contrast with percentile intervals.

    ``reference_index`` defaults to the largest group, which keeps the contrasts as precise
    as the data allows and avoids picking the extreme group *after* looking at the results
    (doing that biases the interval, because the winner of a noisy maximum is optimistic).
    """
    if adjust not in ADJUSTMENTS:
        raise ValueError(f"adjust must be one of {ADJUSTMENTS}, got {adjust!r}")
    methods = choose_methods(counts, interval_method)
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

    resampled = bootstrap_counts(counts, n_boot=n_boot, seed=seed, scheme=scheme).astype(np.float64)
    posterior_groups = [index for index, method in enumerate(methods) if method == DIRICHLET]
    if posterior_groups:
        posterior = posterior_counts(counts, n_boot=n_boot, seed=seed)
        resampled[:, posterior_groups, :] = posterior[:, posterior_groups, :]
    sampled_rates = metrics.per_group_rates(resampled)
    # The prior must not invent a rate the data cannot define (D3): a posterior group whose
    # observed rate is undefined stays undefined in every draw.
    for key, observed in observed_rates.items():
        for index in posterior_groups:
            if not np.isfinite(observed[index]):
                sampled_rates[key][:, index] = np.nan
    sampled_gaps = metrics.compute_gaps(sampled_rates)

    observed_contrasts = _contrasts(observed_rates, reference)
    sampled_contrasts = _contrasts(sampled_rates, reference)
    bands: dict[str, FloatArray] = {}
    family_size = 0
    critical_value = float("nan")
    if adjust == MAX_T:
        bands, family_size, critical_value = _simultaneous_bands(
            observed_contrasts, sampled_contrasts, reference
        )

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
        contrasts=observed_contrasts,
        contrast_ci={
            key: _quantiles(value, quantile_tuple) for key, value in sampled_contrasts.items()
        },
        adjustment=adjust,
        adjusted_contrast_ci=bands,
        family_size=family_size,
        critical_value=critical_value,
        interval_methods=methods,
    )


def _contrasts(rates: Mapping[str, FloatArray], reference: int) -> dict[str, FloatArray]:
    """Signed difference between each group's rate and the reference group's.

    Unlike ``max - min``, this is symmetric about zero under the null, so its interval can
    contain zero and therefore supports an honest "no clear evidence" verdict.
    """
    return {key: value - value[..., reference, None] for key, value in rates.items()}


def _simultaneous_bands(
    observed: Mapping[str, FloatArray],
    sampled: Mapping[str, FloatArray],
    reference: int,
) -> tuple[dict[str, FloatArray], int, float]:
    """Max-t bootstrap bands covering every headline contrast at once (D14).

    Each contrast is standardised by its bootstrap standard deviation. ``T*`` is the 95th
    percentile, over resamples, of the largest standardised deviation from the observed
    value across the whole family. The band for contrast ``j`` is ``obs_j +/- T* sd_j``:
    it is wider than the marginal interval exactly as much as the family needs.

    Contrasts that cannot be standardised are left out of the maximum: undefined ones
    (no finite resamples) get an undefined band, and constant ones (``sd == 0``) keep a
    degenerate band at their observed value. The reference group is zero by construction.
    """
    n_groups = int(observed[VERDICT_RATE_KEYS[0]].shape[-1])
    others = [group for group in range(n_groups) if group != reference]
    bands = {key: np.zeros((2, n_groups)) for key in VERDICT_RATE_KEYS}
    if not others:
        return bands, 0, float("nan")

    obs = np.concatenate([observed[key][others] for key in VERDICT_RATE_KEYS])
    draws = np.concatenate([sampled[key][:, others] for key in VERDICT_RATE_KEYS], axis=1)
    finite = np.isfinite(draws)
    count = finite.sum(axis=0)
    mean = np.where(finite, draws, 0.0).sum(axis=0) / np.maximum(count, 1)
    variance = (np.where(finite, draws - mean, 0.0) ** 2).sum(axis=0) / np.maximum(count - 1, 1)
    sd = np.sqrt(variance)

    defined = np.isfinite(obs) & (count >= 2)
    usable = defined & (sd > 0.0)

    critical = float("nan")
    if usable.any():
        deviation = np.abs(draws[:, usable] - obs[usable]) / sd[usable]
        largest = np.where(finite[:, usable], deviation, -np.inf).max(axis=1)
        largest = largest[np.isfinite(largest)]
        if largest.size:
            critical = float(np.quantile(largest, FAMILY_LEVEL))

    half = np.where(usable, critical * sd, 0.0)
    low = np.where(defined, obs - half, np.nan)
    high = np.where(defined, obs + half, np.nan)
    # A constant contrast has nothing to widen; an unusable one with a NaN critical value
    # (no usable contrasts anywhere) stays degenerate rather than inventing a width.
    if not usable.any():
        low, high = np.where(defined, obs, np.nan), np.where(defined, obs, np.nan)

    n_others = len(others)
    for index, key in enumerate(VERDICT_RATE_KEYS):
        block = slice(index * n_others, (index + 1) * n_others)
        bands[key][0, others] = low[block]
        bands[key][1, others] = high[block]
    return bands, int(usable.sum()), critical
