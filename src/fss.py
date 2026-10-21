"""Finite-size scaling: Binder cumulants, time series and error bars for N = 32..128."""

import argparse
import time

import numpy as np
import torch

from src.cpu_ising import T_C
from src.data_io import save_fss_hdf5
from src.gpu_ising import GPUIsing2D
from src.stats import compute_autocorrelation, jackknife_resampling

OUTPUT_PATH = "data/fss_analysis.h5"
SIZES = (32, 64, 128)


def compute_binder_cumulant(M_series: np.ndarray) -> float:
    """U4 = 1 - <M^4> / (3 <M^2>^2). Valid for signed or absolute M (even moments)."""
    m = np.asarray(M_series, dtype=np.float64)
    return float(1.0 - np.mean(m**4) / (3.0 * np.mean(m**2) ** 2))


def find_crossing(T: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    """T where curves y1 and y2 first cross (linear interpolation), else None."""
    d = np.asarray(y1) - np.asarray(y2)
    for i in np.nonzero(d[:-1] * d[1:] <= 0)[0]:
        if d[i] == d[i + 1]:
            continue
        return float(T[i] + (T[i + 1] - T[i]) * d[i] / (d[i] - d[i + 1]))
    return None


def analyze_series(
    M: np.ndarray, E: np.ndarray, N: int, T: float, n_blocks: int
) -> dict[str, float]:
    """Observables with jackknife errors from one lattice's |M| and E time series."""
    n2 = N * N
    chi = lambda m: n2 / T * (np.mean(m**2) - np.mean(m) ** 2)  # noqa: E731
    cv = lambda e: n2 / T**2 * (np.mean(e**2) - np.mean(e) ** 2)  # noqa: E731
    chi_v, chi_e = jackknife_resampling(M, chi, n_blocks)
    cv_v, cv_e = jackknife_resampling(E, cv, n_blocks)
    u4_v, u4_e = jackknife_resampling(M, compute_binder_cumulant, n_blocks)
    return {
        "magnetization": float(M.mean()),
        "energy": float(E.mean()),
        "susceptibility": chi_v,
        "susceptibility_err": chi_e,
        "specific_heat": cv_v,
        "specific_heat_err": cv_e,
        "binder": u4_v,
        "binder_err": u4_e,
        "tau_int_M": compute_autocorrelation(M)[1],
    }


def run_fss(
    sizes: tuple[int, ...] = SIZES,
    temps: np.ndarray | None = None,
    therm_sweeps: int = 10000,
    meas_sweeps: int = 100000,
    n_blocks: int = 50,
    seed: int = 0,
    device: str = "cuda:0",
) -> dict[int, dict[str, np.ndarray]]:
    """One GPU batch per lattice size (one chain per temperature), series kept per sweep."""
    if temps is None:
        temps = np.round(np.arange(2.15, 2.40 + 1e-9, 0.005), 3)
    B = len(temps)
    results: dict[int, dict[str, np.ndarray]] = {}
    for N in sizes:
        torch.manual_seed(seed + N)
        model = GPUIsing2D(N=N, T=temps, batch=B, init="random", device=device)
        model.spins[torch.as_tensor(temps < T_C, device=model.device)] = 1  # cold start

        t0 = time.perf_counter()
        model.sweep(therm_sweeps)
        M = torch.empty(meas_sweeps, B, dtype=torch.float32, device=model.device)
        E = torch.empty_like(M)
        for t in range(meas_sweeps):
            model.sweep(1)
            M[t] = model.compute_magnetization()
            E[t] = model.compute_energy()
        M64, E64 = M.cpu().numpy().astype(np.float64), E.cpu().numpy().astype(np.float64)

        stats = [analyze_series(M64[:, b], E64[:, b], N, float(temps[b]), n_blocks)
                 for b in range(B)]
        results[N] = {"T": temps.astype(np.float64)}
        for key in stats[0]:
            results[N][key] = np.array([s[key] for s in stats])
        print(f"N={N}: {B} temps, {therm_sweeps}+{meas_sweeps} sweeps, "
              f"{time.perf_counter() - t0:.0f} s, max tau_int(|M|)="
              f"{results[N]['tau_int_M'].max():.0f} sweeps")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-size scaling runs")
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--therm", type=int, default=10000)
    parser.add_argument("--meas", type=int, default=100000)
    parser.add_argument("--blocks", type=int, default=50)
    args = parser.parse_args()

    results = run_fss(therm_sweeps=args.therm, meas_sweeps=args.meas, n_blocks=args.blocks)
    metadata = {"therm_sweeps": args.therm, "meas_sweeps": args.meas,
                "n_blocks": args.blocks, "seed": 0, "T_c_exact": T_C}
    save_fss_hdf5(args.out, metadata, results)

    sizes = sorted(results)
    for a, b in zip(sizes, sizes[1:]):
        Tx = find_crossing(results[a]["T"], results[a]["binder"], results[b]["binder"])
        print(f"U4 crossing N={a}/{b}: " + (f"T = {Tx:.4f}" if Tx else "none in range")
              + f"  (Onsager Tc = {T_C:.4f})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
