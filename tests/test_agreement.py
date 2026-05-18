"""Tests for the agreement module.

Each primitive is validated against an external reference implementation:
- Cohen's κ  -> sklearn.metrics.cohen_kappa_score
- Fleiss' κ  -> statsmodels.stats.inter_rater.fleiss_kappa
- Krippendorff's α -> krippendorff PyPI package

Tolerance: 1e-9 absolute. CI gates on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.agreement import cohens_kappa, fleiss_kappa, krippendorff_alpha

# ---------- Cohen's kappa ----------


def test_cohens_kappa_perfect_agreement() -> None:
    a = np.array([0, 1, 1, 0, 1])
    b = np.array([0, 1, 1, 0, 1])
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_chance_agreement_is_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=1000)
    b = rng.integers(0, 2, size=1000)
    # Random coin-flips ≈ chance agreement -> κ ≈ 0.
    assert abs(cohens_kappa(a, b)) < 0.1


def test_cohens_kappa_matches_sklearn() -> None:
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, size=200)
    b = (a + rng.integers(-1, 2, size=200)).clip(0, 3)
    ours = cohens_kappa(a, b)
    ref = float(cohen_kappa_score(a, b))
    assert abs(ours - ref) < 1e-9


def test_cohens_kappa_string_labels() -> None:
    a = np.array(["yes", "no", "yes", "no", "yes"])
    b = np.array(["yes", "no", "yes", "yes", "yes"])
    # 1 disagreement out of 5; not zero.
    k = cohens_kappa(a, b)
    assert 0.0 < k < 1.0


def test_cohens_kappa_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(np.array([0, 1]), np.array([0]))


def test_cohens_kappa_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cohens_kappa(np.array([]), np.array([]))


def test_cohens_kappa_all_same_category() -> None:
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 1])
    # Degenerate but well-defined: perfect agreement, κ = 1.0.
    assert cohens_kappa(a, b) == pytest.approx(1.0)


# ---------- Fleiss' kappa ----------


def test_fleiss_kappa_perfect_agreement() -> None:
    # 3 items, 5 raters, 2 categories; all raters agree on each item.
    matrix = np.array(
        [
            [5, 0],
            [0, 5],
            [5, 0],
        ],
        dtype=float,
    )
    assert fleiss_kappa(matrix) == pytest.approx(1.0)


def test_fleiss_kappa_matches_statsmodels() -> None:
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss

    # 10 items, 6 raters, 3 categories (Fleiss' 1971 example shape).
    matrix = np.array(
        [
            [0, 0, 6],
            [0, 3, 3],
            [0, 1, 5],
            [0, 0, 6],
            [0, 3, 3],
            [4, 2, 0],
            [3, 2, 1],
            [5, 1, 0],
            [0, 5, 1],
            [3, 0, 3],
        ],
        dtype=float,
    )
    ours = fleiss_kappa(matrix)
    ref = float(sm_fleiss(matrix))
    assert abs(ours - ref) < 1e-9


def test_fleiss_kappa_rejects_inconsistent_rater_counts() -> None:
    matrix = np.array([[2, 1], [3, 1]], dtype=float)
    with pytest.raises(ValueError, match="same number of raters"):
        fleiss_kappa(matrix)


def test_fleiss_kappa_rejects_single_rater() -> None:
    matrix = np.array([[1, 0], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="at least 2 raters"):
        fleiss_kappa(matrix)


def test_fleiss_kappa_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fleiss_kappa(np.array([1, 2, 3], dtype=float))


# ---------- Krippendorff's alpha ----------


def test_krippendorff_alpha_perfect_agreement_nominal() -> None:
    # 3 raters x 5 units; everyone agrees.
    data = np.array(
        [
            [1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2],
        ],
        dtype=float,
    )
    assert krippendorff_alpha(data, level_of_measurement="nominal") == pytest.approx(1.0)


def test_krippendorff_alpha_matches_reference_nominal() -> None:
    import krippendorff as ref_kr

    # Common test fixture from Krippendorff 2011 Section 7.
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
        ],
        dtype=float,
    )
    ours = krippendorff_alpha(data, level_of_measurement="nominal")
    ref = float(ref_kr.alpha(reliability_data=data, level_of_measurement="nominal"))
    assert abs(ours - ref) < 1e-9


def test_krippendorff_alpha_matches_reference_ordinal() -> None:
    import krippendorff as ref_kr

    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5],
            [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5],
        ],
        dtype=float,
    )
    ours = krippendorff_alpha(data, level_of_measurement="ordinal")
    ref = float(ref_kr.alpha(reliability_data=data, level_of_measurement="ordinal"))
    assert abs(ours - ref) < 1e-9


def test_krippendorff_alpha_interval() -> None:
    import krippendorff as ref_kr

    data = np.array(
        [
            [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 3.0, 2.0, 2.0],
            [1.0, 3.0, 3.0, 3.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    ours = krippendorff_alpha(data, level_of_measurement="interval")
    ref = float(ref_kr.alpha(reliability_data=data, level_of_measurement="interval"))
    assert abs(ours - ref) < 1e-9


def test_krippendorff_alpha_rejects_unknown_level() -> None:
    data = np.array([[1, 2], [1, 2]], dtype=float)
    with pytest.raises(ValueError, match="level_of_measurement"):
        krippendorff_alpha(data, level_of_measurement="banana")


def test_krippendorff_alpha_rejects_single_rater() -> None:
    data = np.array([[1, 2, 3]], dtype=float)
    with pytest.raises(ValueError, match="at least 2 raters"):
        krippendorff_alpha(data)


def test_krippendorff_alpha_rejects_all_missing() -> None:
    data = np.full((2, 4), np.nan)
    with pytest.raises(ValueError, match="non-missing"):
        krippendorff_alpha(data)
