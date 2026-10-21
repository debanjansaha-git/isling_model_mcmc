import numpy as np
import pytest
import torch

from src.data_io import load_fss_hdf5, save_fss_hdf5
from src.fss import compute_binder_cumulant, find_crossing, run_fss
from src.stats import compute_autocorrelation, jackknife_resampling


def ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    """AR(1) series x_t = phi x_{t-1} + eps_t with unit-variance innovations."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


# ---- autocorrelation -------------------------------------------------------

def test_autocorrelation_white_noise() -> None:
    C, tau = compute_autocorrelation(np.random.default_rng(1).standard_normal(50_000))
    assert C[0] == pytest.approx(1.0)
    assert tau == pytest.approx(0.5, abs=0.15)


def test_autocorrelation_ar1_matches_analytic_tau() -> None:
    phi = 0.9
    C, tau = compute_autocorrelation(ar1(200_000, phi))
    assert C[1] == pytest.approx(phi, abs=0.02)
    assert tau == pytest.approx((1 + phi) / (2 * (1 - phi)), rel=0.2)  # = 9.5


def test_autocorrelation_constant_series_is_finite() -> None:
    C, tau = compute_autocorrelation(np.ones(100))
    assert C[0] == 1.0 and np.isfinite(tau)


# ---- jackknife -------------------------------------------------------------

def test_delete_one_jackknife_of_mean_equals_standard_error() -> None:
    x = np.random.default_rng(2).standard_normal(300)
    est, err = jackknife_resampling(x, np.mean, n_blocks=len(x))
    assert est == pytest.approx(x.mean())
    assert err == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)))


def test_jackknife_removes_bias_of_variance() -> None:
    x = np.random.default_rng(3).standard_normal(40)
    est, _ = jackknife_resampling(x, np.var, n_blocks=len(x))  # np.var is biased (ddof=0)
    assert est == pytest.approx(x.var(ddof=1))


def test_blocked_jackknife_error_on_correlated_noise() -> None:
    phi, n = 0.9, 200_000
    x = ar1(n, phi, seed=4)
    sigma2 = 1 / (1 - phi**2)
    tau = (1 + phi) / (2 * (1 - phi))
    true_err = np.sqrt(2 * tau * sigma2 / n)

    _, err_blocked = jackknife_resampling(x, np.mean, n_blocks=50)
    _, err_naive = jackknife_resampling(x, np.mean, n_blocks=n)
    assert err_blocked == pytest.approx(true_err, rel=0.35)
    assert err_naive < 0.5 * true_err  # ignoring correlations underestimates the error


def test_jackknife_rejects_too_few_blocks() -> None:
    with pytest.raises(ValueError):
        jackknife_resampling(np.arange(10.0), np.mean, n_blocks=1)


# ---- Binder cumulant -------------------------------------------------------

def test_binder_cumulant_gaussian_is_zero() -> None:
    # <M^4> = 3 <M^2>^2 for Gaussian M, so U4 -> 0 (disordered limit).
    M = np.random.default_rng(5).standard_normal(2_000_000)
    assert compute_binder_cumulant(M) == pytest.approx(0.0, abs=0.01)


def test_binder_cumulant_ordered_is_two_thirds() -> None:
    M = np.random.default_rng(6).choice([-1.0, 1.0], size=1000)
    assert compute_binder_cumulant(M) == pytest.approx(2 / 3)
    assert compute_binder_cumulant(np.abs(M)) == pytest.approx(2 / 3)


def test_find_crossing_linear_interpolation() -> None:
    T = np.array([1.0, 2.0, 3.0])
    assert find_crossing(T, T, 4.0 - T) == pytest.approx(2.0)
    assert find_crossing(T, T, T + 1.0) is None


# ---- IO + runner -----------------------------------------------------------

def test_fss_hdf5_roundtrip(tmp_path) -> None:
    res = {16: {"T": np.linspace(2, 3, 5), "binder": np.linspace(0.6, 0.1, 5)},
           32: {"T": np.linspace(2, 3, 5), "binder": np.linspace(0.65, 0.05, 5)}}
    save_fss_hdf5(tmp_path / "fss.h5", {"seed": 7}, res)
    out = load_fss_hdf5(tmp_path / "fss.h5")
    assert out["metadata"]["seed"] == 7
    assert sorted(out["results"]) == [16, 32]
    np.testing.assert_allclose(out["results"][32]["binder"], res[32]["binder"], rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_fss_small_run_is_physical() -> None:
    temps = np.array([1.5, 4.0])
    res = run_fss(sizes=(16,), temps=temps, therm_sweeps=200, meas_sweeps=2000,
                  n_blocks=10, seed=0)
    r = res[16]
    for key in ("T", "binder", "binder_err", "susceptibility", "susceptibility_err",
                "specific_heat", "specific_heat_err", "magnetization", "energy", "tau_int_M"):
        assert r[key].shape == (2,)
    assert r["binder"][0] == pytest.approx(2 / 3, abs=0.02)  # ordered
    assert r["binder"][1] < 0.2  # disordered
