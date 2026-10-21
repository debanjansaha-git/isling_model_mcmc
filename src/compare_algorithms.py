"""Metropolis vs Wolff integrated autocorrelation times near T_c."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.gpu_ising import GPUIsing2D
from src.stats import compute_autocorrelation
from src.wolff_ising import WolffIsing2D

T_CRIT = 2.269
SIZES = (16, 32, 64, 128)
OUTPUT_JSON = "data/compare_algorithms.json"
OUTPUT_PNG = "docs/images/critical_slowing_down.png"


def metropolis_tau(N: int, chains: int, therm: int, meas: int, seed: int) -> tuple[float, float]:
    """tau_int(|M|) in sweeps: mean and standard error over independent GPU chains."""
    torch.manual_seed(seed)
    model = GPUIsing2D(N=N, T=T_CRIT, batch=chains, init="up")  # T < T_c: cold start
    model.sweep(therm)
    series = torch.empty(meas, chains, dtype=torch.float32, device=model.device)
    for t in range(meas):
        model.sweep(1)
        series[t] = model.compute_magnetization()
    series = series.cpu().numpy().astype(np.float64)
    taus = np.array([compute_autocorrelation(series[:, c])[1] for c in range(chains)])
    return float(taus.mean()), float(taus.std(ddof=1) / np.sqrt(chains))


def wolff_tau(N: int, updates: int, therm: int, seed: int) -> dict[str, float]:
    """tau_int(|M|) in cluster updates, and converted to sweep equivalents.

    One Wolff update visits <|C|> spins, so it costs <|C|>/N^2 of a sweep.
    """
    np.random.seed(seed)
    model = WolffIsing2D(N=N, T=T_CRIT, init="up")
    model.run(therm)
    mags, sizes = model.run(updates)
    tau = compute_autocorrelation(mags)[1]
    frac = float(sizes.mean() / N**2)
    return {"tau_cluster": float(tau), "cluster_fraction": frac, "tau_sweeps": float(tau * frac)}


def fit_exponent(sizes: list[int], taus: list[float]) -> float:
    return float(np.polyfit(np.log(sizes), np.log(taus), 1)[0])


def plot(res: dict, out: str = OUTPUT_PNG) -> None:
    sizes = res["sizes"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sizes, res["metropolis"]["tau"], yerr=res["metropolis"]["tau_err"], fmt="o-",
                color="tab:red", capsize=3,
                label=f"Metropolis (sweeps), z = {res['z_metropolis']:.2f}")
    ax.plot(sizes, res["wolff"]["tau_cluster"], "s-", color="tab:blue",
            label=f"Wolff (cluster updates), z = {res['z_wolff_cluster']:.2f}")
    ax.plot(sizes, res["wolff"]["tau_sweeps"], "^--", color="tab:cyan",
            label=f"Wolff (sweep-equivalents), z = {res['z_wolff_sweeps']:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlabel("Lattice size N")
    ax.set_ylabel(r"$\tau_{int}(|M|)$")
    ax.set_title(rf"Critical slowing down at T = {T_CRIT} ($\tau \propto N^z$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Metropolis vs Wolff autocorrelation")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--metro-therm", type=int, default=10000)
    parser.add_argument("--wolff-updates", type=int, default=20000)
    parser.add_argument("--out", default=OUTPUT_JSON)
    args = parser.parse_args()

    sizes = list(SIZES)
    metro = {"tau": [], "tau_err": []}
    wolff = {"tau_cluster": [], "cluster_fraction": [], "tau_sweeps": []}
    for N in sizes:
        meas = 50_000 if N <= 32 else 100_000
        t0 = time.perf_counter()
        tau, err = metropolis_tau(N, args.chains, args.metro_therm, meas, seed=N)
        t1 = time.perf_counter()
        w = wolff_tau(N, args.wolff_updates, therm=1000, seed=N)
        t2 = time.perf_counter()
        metro["tau"].append(tau)
        metro["tau_err"].append(err)
        for k in wolff:
            wolff[k].append(w[k])
        print(f"N={N}: Metropolis tau={tau:.1f}±{err:.1f} sweeps ({t1 - t0:.0f} s) | "
              f"Wolff tau={w['tau_cluster']:.2f} updates, <C>/N^2={w['cluster_fraction']:.3f}, "
              f"= {w['tau_sweeps']:.2f} sweeps ({t2 - t1:.0f} s)")

    res = {
        "T": T_CRIT, "sizes": sizes, "chains": args.chains,
        "metropolis": metro, "wolff": wolff,
        "wolff_updates": args.wolff_updates, "metro_therm": args.metro_therm,
        "z_metropolis": fit_exponent(sizes, metro["tau"]),
        "z_wolff_cluster": fit_exponent(sizes, wolff["tau_cluster"]),
        "z_wolff_sweeps": fit_exponent(sizes, wolff["tau_sweeps"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    plot(res)
    print(f"z (Metropolis) = {res['z_metropolis']:.2f}, z (Wolff, cluster updates) = "
          f"{res['z_wolff_cluster']:.2f}, z (Wolff, sweep-equiv) = {res['z_wolff_sweeps']:.2f}")
    print(f"saved {args.out} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
