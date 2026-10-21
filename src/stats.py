"""Autocorrelation and jackknife error analysis for correlated MCMC series."""

from typing import Callable

import numpy as np


def compute_autocorrelation(
    series: np.ndarray, window_c: float = 6.0
) -> tuple[np.ndarray, float]:
    """Normalized autocorrelation C(t) (C(0)=1) and integrated time tau_int.

    tau_int = 1/2 + sum_{t=1}^{W} C(t) (so white noise gives 1/2 and the variance
    of the mean is 2 tau_int sigma^2 / n). The window W is chosen automatically
    as the smallest W >= window_c * tau_int(W) (Sokal's criterion).
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)  # zero-pad to avoid circular wrap-around
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0.0:  # constant series
        C = np.zeros(n)
        C[0] = 1.0
        return C, 0.5
    C = acf / acf[0]
    tau_w = 0.5 + np.cumsum(C[1:])  # tau_int(W) for W = 1..n-1
    ok = np.nonzero(np.arange(1, n) >= window_c * tau_w)[0]
    W = ok[0] if len(ok) else n - 2
    return C, float(tau_w[W])


def jackknife_resampling(
    series: np.ndarray,
    func: Callable[[np.ndarray], float],
    n_blocks: int = 50,
) -> tuple[float, float]:
    """Blocked delete-one jackknife of func(series).

    Returns (bias-corrected estimate, standard error). Blocks must be longer than
    the autocorrelation time for the error to account for correlations; with
    n_blocks == len(series) this is the ordinary delete-one jackknife.
    """
    x = np.asarray(series)
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    size = len(x) // n_blocks
    if size < 1:
        raise ValueError("series shorter than n_blocks")
    x = x[: size * n_blocks]
    blocks = x.reshape(n_blocks, size)

    theta_full = float(func(x))
    theta_i = np.array(
        [func(np.delete(blocks, i, axis=0).ravel()) for i in range(n_blocks)]
    )
    theta_bar = theta_i.mean()
    estimate = n_blocks * theta_full - (n_blocks - 1) * theta_bar
    error = np.sqrt((n_blocks - 1) / n_blocks * np.sum((theta_i - theta_bar) ** 2))
    return float(estimate), float(error)
