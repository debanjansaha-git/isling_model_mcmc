"""Production temperature sweep: all Markov chains run in one GPU batch."""

import argparse
import time

import numpy as np
import torch

from src.cpu_ising import T_C
from src.data_io import save_simulation_hdf5
from src.gpu_ising import GPUIsing2D

OUTPUT_PATH = "data/ising_sweep_N128.h5"


def run_simulation(
    N: int = 128,
    temps: np.ndarray | None = None,
    therm_sweeps: int = 3000,
    meas_sweeps: int = 10000,
    seed: int = 0,
    device: str = "cuda:0",
) -> dict:
    """Thermalize, then measure every sweep. Returns scalars, snapshots, run stats."""
    if temps is None:
        temps = np.round(np.arange(1.0, 4.0 + 1e-9, 0.02), 2)
    B = len(temps)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Cold start below Tc (hot starts freeze into stripe domains), hot start above.
    model = GPUIsing2D(N=N, T=temps, batch=B, init="random", device=device)
    cold = torch.as_tensor(temps < T_C, device=model.device)
    model.spins[cold] = 1

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.sweep(therm_sweeps)

    z = lambda: torch.zeros(B, dtype=torch.float64, device=model.device)  # noqa: E731
    s_absM, s_M2, s_E, s_E2 = z(), z(), z(), z()
    for _ in range(meas_sweeps):
        model.sweep(1)
        m = model.compute_magnetization()  # |M| per spin
        e = model.compute_energy()
        s_absM += m
        s_M2 += m * m
        s_E += e
        s_E2 += e * e

    torch.cuda.synchronize()
    runtime = time.perf_counter() - t0

    absM, M2 = s_absM / meas_sweeps, s_M2 / meas_sweeps
    E, E2 = s_E / meas_sweeps, s_E2 / meas_sweeps
    T = model.T.reshape(B).double()
    n_spins = N * N
    scalars = {
        "T": temps.astype(np.float64),
        "magnetization": absM.cpu().numpy(),
        "energy": E.cpu().numpy(),
        "susceptibility": (n_spins / T * (M2 - absM**2)).cpu().numpy(),
        "specific_heat": (n_spins / T**2 * (E2 - E**2)).cpu().numpy(),
    }
    total_sweeps = therm_sweeps + meas_sweeps
    stats = {
        "N": N,
        "batch": B,
        "therm_sweeps": therm_sweeps,
        "meas_sweeps": meas_sweeps,
        "seed": seed,
        "runtime_s": runtime,
        "spin_updates_per_s": B * n_spins * total_sweeps / runtime,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu": torch.cuda.get_device_name(model.device),
        "T_c_exact": T_C,
    }
    return {"scalars": scalars, "snapshots": model.spins.cpu().numpy(), "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="N=128 Ising temperature sweep")
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--therm", type=int, default=3000)
    parser.add_argument("--meas", type=int, default=10000)
    args = parser.parse_args()

    res = run_simulation(therm_sweeps=args.therm, meas_sweeps=args.meas)
    st, sc = res["stats"], res["scalars"]
    save_simulation_hdf5(args.out, st, sc, res["snapshots"])

    print(f"{st['batch']} lattices, N={st['N']}, "
          f"{st['therm_sweeps'] + st['meas_sweeps']} sweeps in {st['runtime_s']:.1f} s")
    print(f"{st['spin_updates_per_s']:.3e} spin-updates/s, "
          f"peak VRAM {st['peak_vram_mib']:.0f} MiB")
    print(f"chi peak at T={sc['T'][np.argmax(sc['susceptibility'])]:.2f}, "
          f"Cv peak at T={sc['T'][np.argmax(sc['specific_heat'])]:.2f} "
          f"(Onsager Tc={T_C:.4f})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
