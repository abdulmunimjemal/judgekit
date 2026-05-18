"""Inter-rater agreement metrics for multi-judge / multi-annotator setups.

When your calibration anchors come from multiple humans (or multiple LLM
judges), you should know how much the raters agree before you trust the
labels. Low agreement on the gold set means your ground truth is noisy
and any calibrator you fit on it will inherit that noise.

Three classic primitives ship here:

- :func:`cohens_kappa` — two raters, categorical labels. The basic
  agreement metric for paired ratings.
- :func:`fleiss_kappa` — N ≥ 2 raters, multi-category. Same item must
  be rated by exactly N raters; categories must be discrete.
- :func:`krippendorff_alpha` — N raters, multi-category, handles
  missing values, supports nominal / ordinal / interval / ratio levels.
  The most flexible of the three; the de facto standard in modern eval
  literature.

All three return a float in (-∞, 1]. Standard rule-of-thumb thresholds
from Landis & Koch (1977) for κ:

- 0.81 - 1.00: almost perfect agreement
- 0.61 - 0.80: substantial
- 0.41 - 0.60: moderate
- 0.21 - 0.40: fair
- 0.00 - 0.20: slight
- < 0.00: less than chance (rare; raters systematically disagree)

Math references:
- Cohen (1960), "A Coefficient of Agreement for Nominal Scales".
- Fleiss (1971), "Measuring nominal scale agreement among many raters".
- Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability".
- Hayes & Krippendorff (2007), "Answering the call for a standard
  reliability measure for coding data".
"""

from __future__ import annotations

import numpy as np


def cohens_kappa(rater_a: np.ndarray, rater_b: np.ndarray) -> float:
    """Cohen's κ for two raters, discrete categorical labels.

    Both inputs must be 1-D arrays of the same length. Labels can be
    any hashable type (ints, strings, ...). Returns a float in (-∞, 1];
    1.0 = perfect agreement, 0.0 = chance, < 0 = worse than chance.
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("rater_a and rater_b must be 1-D arrays of the same length")
    if a.size == 0:
        raise ValueError("rater arrays must be non-empty")

    categories = np.unique(np.concatenate([a, b]))
    k = len(categories)
    confusion = np.zeros((k, k), dtype=float)
    cat_index = {c: i for i, c in enumerate(categories)}
    for x, y in zip(a, b, strict=True):
        confusion[cat_index[x], cat_index[y]] += 1.0
    n = confusion.sum()

    p_o = float(np.trace(confusion) / n)
    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    p_e = float(np.sum(row_marginals * col_marginals))
    if abs(1.0 - p_e) < 1e-12:
        # All raters agreed on the same single category. κ is technically
        # undefined; return 1.0 if observed agreement is also perfect,
        # else 0.0 (no information beyond chance).
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' κ for N ≥ 2 raters, M items, K discrete categories.

    Input is a (M, K) matrix where ``ratings[i, j]`` is the number of
    raters who assigned item ``i`` to category ``j``. Every row must
    sum to the same N (the number of raters).

    Returns a float in (-∞, 1].
    """
    matrix = np.asarray(ratings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("ratings must be a 2-D matrix (items x categories)")
    m, _k = matrix.shape
    if m < 1:
        raise ValueError("ratings must have at least one item")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, row_sums[0]):
        raise ValueError(
            "every item must be rated by exactly the same number of raters "
            f"(got per-row sums {row_sums.tolist()})"
        )
    n_raters = float(row_sums[0])
    if n_raters < 2:
        raise ValueError("Fleiss' kappa requires at least 2 raters per item")

    # P_i — proportion of agreeing pairs per item.
    p_i = (np.sum(matrix * matrix, axis=1) - n_raters) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_i))

    # P_e — expected agreement from category prevalence.
    column_marginals = matrix.sum(axis=0) / (m * n_raters)
    p_e = float(np.sum(column_marginals**2))

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def _pair_distance(
    c_idx: int,
    k_idx: int,
    values: np.ndarray,
    n_c: np.ndarray,
    level: str,
) -> float:
    """Krippendorff's per-pair distance metric.

    Level definitions follow Krippendorff (2011), Section 6:
    - ``nominal``: 0 if categories equal, 1 otherwise.
    - ``ordinal``: sum-of-marginals between categories, with half the
      endpoint marginals subtracted, squared. Depends on the global
      marginal counts ``n_c``.
    - ``interval``: ``(values[c] - values[k])²``.
    - ``ratio``: ``((values[c] - values[k]) / (values[c] + values[k]))²``.
    """
    c_val = float(values[c_idx])
    k_val = float(values[k_idx])
    if level == "nominal":
        return 0.0 if c_idx == k_idx else 1.0
    if level == "ordinal":
        # Sum of marginal counts strictly between c_idx and k_idx,
        # plus half of n_c[c_idx] and half of n_c[k_idx]. Then square.
        lo = min(c_idx, k_idx)
        hi = max(c_idx, k_idx)
        between_sum = float(n_c[lo + 1 : hi].sum()) if hi - lo >= 2 else 0.0
        delta = float(n_c[lo]) / 2.0 + between_sum + float(n_c[hi]) / 2.0
        return float(delta**2)
    if level == "interval":
        return float((c_val - k_val) ** 2)
    if level == "ratio":
        denom = c_val + k_val
        if abs(denom) < 1e-12:
            return 0.0
        return float(((c_val - k_val) / denom) ** 2)
    raise ValueError(f"unknown level_of_measurement: {level!r}")


def krippendorff_alpha(
    reliability_data: np.ndarray,
    level_of_measurement: str = "nominal",
) -> float:
    """Krippendorff's α for N raters, M units, with optional missing values.

    ``reliability_data`` is a (N_raters, M_units) array. Missing values
    are represented by ``np.nan``. Levels: ``"nominal"``, ``"ordinal"``,
    ``"interval"``, ``"ratio"``.

    Returns a float in (-∞, 1]. By convention α >= 0.80 = reliable;
    0.667 = tentative; < 0.667 = unreliable.

    Algorithm follows Krippendorff (2011) "Computing Krippendorff's
    Alpha-Reliability" Section 7: builds the coincidence matrix, the
    expected-coincidence matrix, then α = 1 - D_o / D_e where D_o and
    D_e are weighted disagreement sums.
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2-D (raters x units)")
    n_raters, n_units = data.shape
    if n_raters < 2:
        raise ValueError("need at least 2 raters")

    # Collect the unique non-NaN values; ordinal/ratio/interval need ordering.
    valid_mask = ~np.isnan(data)
    if not valid_mask.any():
        raise ValueError("reliability_data has no non-missing values")
    values = np.unique(data[valid_mask])
    n_values = len(values)
    value_index = {v: i for i, v in enumerate(values)}

    # Coincidence matrix: o[c, k] = number of c-k coincidences across
    # all units, summed over all rater pairs and weighted so each unit
    # contributes proportional to 1 / (m_u - 1) where m_u is the number
    # of raters who scored that unit.
    o = np.zeros((n_values, n_values), dtype=float)
    n_c = np.zeros(n_values, dtype=float)  # marginal counts
    for u in range(n_units):
        col = data[:, u]
        present = col[~np.isnan(col)]
        m_u = len(present)
        if m_u < 2:
            continue
        for x in present:
            for y in present:
                if x == y:
                    continue
                o[value_index[x], value_index[y]] += 1.0 / (m_u - 1)
        for x in present:
            n_c[value_index[x]] += 1.0

    n_total = n_c.sum()
    if n_total < 2:
        raise ValueError("not enough pairable observations")

    # Expected coincidence matrix.
    e = np.zeros_like(o)
    for c_idx in range(n_values):
        for k_idx in range(n_values):
            if c_idx == k_idx:
                e[c_idx, k_idx] = n_c[c_idx] * (n_c[c_idx] - 1) / (n_total - 1)
            else:
                e[c_idx, k_idx] = n_c[c_idx] * n_c[k_idx] / (n_total - 1)

    d_o = 0.0
    d_e = 0.0
    for c_idx in range(n_values):
        for k_idx in range(n_values):
            if c_idx >= k_idx:
                continue
            dist = _pair_distance(c_idx, k_idx, values, n_c, level_of_measurement)
            d_o += o[c_idx, k_idx] * dist
            d_e += e[c_idx, k_idx] * dist

    if abs(d_e) < 1e-12:
        # Degenerate: no expected disagreement, so we're either perfect or
        # the data has no information. Return 1.0 if d_o is also ~0.
        return 1.0 if abs(d_o) < 1e-12 else 0.0
    return 1.0 - d_o / d_e
