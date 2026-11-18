import numpy as np

from experiments.real_data_als_optimization_targets import extract_targets
from experiments.real_data_als_three_target_replay import normalize_objective


SETTINGS = {
    "savgol_window_points": 11,
    "savgol_polynomial_order": 3,
    "peak_minimum_distance_points": 5,
    "peak_prominence_error_multiplier": 5.0,
    "peak_prominence_robust_range_fraction": 0.01,
}


def test_gaussian_target_extraction_recovers_peak_and_fwhm():
    q = np.linspace(0.0, 10.0, 1001)
    sigma = 0.4
    intensity = 3.0 + 12.0 * np.exp(-0.5 * ((q - 4.0) / sigma) ** 2)
    result = extract_targets(q, intensity, np.full_like(q, 0.01), SETTINGS)
    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert result["peak_valid"] == 1
    assert abs(result["peak_q"] - 4.0) < 0.02
    assert abs(result["maximum_peak_height"] - 15.0) < 0.05
    assert abs(result["fwhm"] - expected_fwhm) < 0.03


def test_monotone_profile_is_flagged_instead_of_inventing_a_peak():
    q = np.linspace(0.0, 10.0, 101)
    intensity = 20.0 - q
    result = extract_targets(q, intensity, np.full_like(q, 0.01), SETTINGS)
    assert result["peak_valid"] == 0
    assert np.isnan(result["fwhm"])
    assert np.isfinite(result["integrated_intensity"])


def test_objective_normalization_respects_direction():
    values = np.asarray([2.0, 4.0, 6.0])
    assert np.allclose(normalize_objective(values, "maximize"), [0.0, 0.5, 1.0])
    assert np.allclose(normalize_objective(values, "minimize"), [1.0, 0.5, 0.0])
