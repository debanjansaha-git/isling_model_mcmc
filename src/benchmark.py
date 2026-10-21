"""Benchmark CPU vs GPU Ising throughput; writes BENCHMARK_REPORT.md and a chart."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.cpu_ising import CPUIsing2D
from src.gpu_ising import GPUIsing2D

LATTICES = [32, 64, 128, 256, 512]
BATCHES = [1, 100, 1000, 10000]
SWEEPS_FULL = 1000
VRAM_LIMIT_BYTES = 1 << 30  # int8 lattice storage must stay below 1 GiB
TIME_BUDGET = 1.0  # seconds spent timing each configuration


def bench_cpu(N: int) -> tuple[float, bool]:
    """Seconds for SWEEPS_FULL sweeps; True if measured fully, False if extrapolated."""
    model = CPUIsing2D(N=N, T=2.269, init="random")
    model.metropolis_sweep(1)  # warm-up
    done, t0 = 0, time.perf_counter()
    while done < SWEEPS_FULL and time.perf_counter() - t0 < TIME_BUDGET * 3:
        model.metropolis_sweep(1)
        done += 1
    dt = time.perf_counter() - t0
    return dt * SWEEPS_FULL / done, done == SWEEPS_FULL


def bench_gpu(N: int, B: int, sweeps: int | None = None) -> tuple[float, int]:
    """Returns (seconds per sweep, sweeps timed). sweeps=None -> time budget."""
    torch.manual_seed(0)
    T = torch.linspace(1.0, 4.0, B)
    model = GPUIsing2D(N=N, T=T, batch=B, init="random")
    model.sweep(2)  # warm-up
    torch.cuda.synchronize()
    if sweeps is None:
        t0 = time.perf_counter()
        model.sweep(1)
        torch.cuda.synchronize()
        one = time.perf_counter() - t0
        sweeps = max(3, min(1000, int(TIME_BUDGET / max(one, 1e-6))))
    t0 = time.perf_counter()
    model.sweep(sweeps)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return dt / sweeps, sweeps


def fmt_time(s: float) -> str:
    return f"{s:.2f} s" if s < 600 else f"{s / 3600:.2f} h"


def main() -> None:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    cpu: dict[int, tuple[float, bool]] = {}
    for N in LATTICES:
        cpu[N] = bench_cpu(N)
        print(f"CPU N={N}: {cpu[N][0]:.2f} s / {SWEEPS_FULL} sweeps"
              f"{'' if cpu[N][1] else ' (extrapolated)'}")

    grid: dict[tuple[int, int], float | None] = {}  # (N, B) -> seconds per sweep
    for N in LATTICES:
        for B in BATCHES:
            if B * N * N > VRAM_LIMIT_BYTES:
                grid[(N, B)] = None
                print(f"GPU N={N} B={B}: skipped (>1 GiB lattice storage)")
                continue
            spp, n = bench_gpu(N, B)
            grid[(N, B)] = spp
            print(f"GPU N={N} B={B}: {1 / spp:.1f} sweeps/s ({B * N * N / spp:.2e} flips/s)")

    gpu_1000: dict[int, float] = {}
    for N in LATTICES:
        spp, _ = bench_gpu(N, 1, sweeps=SWEEPS_FULL)
        gpu_1000[N] = spp * SWEEPS_FULL
        print(f"GPU N={N} B=1: {gpu_1000[N]:.2f} s / {SWEEPS_FULL} sweeps")

    write_report(gpu_name, cpu, grid, gpu_1000)
    write_chart(cpu, grid)
    print("Wrote BENCHMARK_REPORT.md and benchmark_results.png")


def write_report(
    gpu_name: str,
    cpu: dict[int, tuple[float, bool]],
    grid: dict[tuple[int, int], float | None],
    gpu_1000: dict[int, float],
) -> None:
    L = ["# Benchmark Report", "",
         f"- GPU: {gpu_name}, torch {torch.__version__}, CUDA {torch.version.cuda}",
         "- CPU: pure-Python single-spin Metropolis (`CPUIsing2D`), one lattice",
         "- GPU: checkerboard engine, int8 spins, per-lattice temperatures T in [1, 4]",
         "- One *sweep* = N² spin-update proposals per lattice. Sweeps/s below are "
         "batch sweeps (all B lattices advance together).",
         "- Configurations whose int8 lattice storage exceeds 1 GiB are skipped (`—`).", ""]

    def table(title: str, cell) -> None:
        L.extend([f"## {title}", "",
                  "| N \\ B | " + " | ".join(str(b) for b in BATCHES) + " |",
                  "|---|" + "---|" * len(BATCHES)])
        for N in LATTICES:
            row = [cell(grid[(N, B)], N, B) for B in BATCHES]
            L.append(f"| {N} | " + " | ".join(row) + " |")
        L.append("")

    table("GPU throughput: sweeps/s", lambda s, N, B: "—" if s is None else f"{1 / s:,.1f}")
    table("GPU throughput: spin-flips/s",
          lambda s, N, B: "—" if s is None else f"{B * N * N / s:.2e}")

    L.extend([f"## CPU vs GPU: {SWEEPS_FULL} sweeps, single lattice (B=1)", "",
              "| N | CPU time | GPU time | Speedup | CPU flips/s | GPU flips/s |",
              "|---|---|---|---|---|---|"])
    for N in LATTICES:
        c, measured = cpu[N]
        g = gpu_1000[N]
        flips = N * N * SWEEPS_FULL
        L.append(f"| {N} | {fmt_time(c)}{'' if measured else ' *'} | {fmt_time(g)} | "
                 f"{c / g:,.1f}x | {flips / c:.2e} | {flips / g:.2e} |")
    L.extend(["", "\\* extrapolated from a timed run of fewer than 1000 sweeps.", ""])

    L.extend([f"## CPU vs GPU: best-case throughput (largest feasible batch)", "",
              "| N | Best B | GPU flips/s | CPU flips/s | Speedup |", "|---|---|---|---|---|"])
    for N in LATTICES:
        B = max(b for b in BATCHES if grid[(N, b)] is not None)
        gf = B * N * N / grid[(N, B)]
        cf = N * N * SWEEPS_FULL / cpu[N][0]
        L.append(f"| {N} | {B} | {gf:.2e} | {cf:.2e} | {gf / cf:,.0f}x |")
    L.append("")
    with open("BENCHMARK_REPORT.md", "w") as f:
        f.write("\n".join(L))


def write_chart(
    cpu: dict[int, tuple[float, bool]], grid: dict[tuple[int, int], float | None]
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(LATTICES, [SWEEPS_FULL / cpu[N][0] for N in LATTICES],
            "k--o", label="CPU (B=1)")
    for B in BATCHES:
        pts = [(N, 1 / grid[(N, B)]) for N in LATTICES if grid[(N, B)] is not None]
        ax.plot(*zip(*pts), "-o", label=f"GPU (B={B})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(LATTICES, [str(n) for n in LATTICES])
    ax.set_xlabel("Lattice size N")
    ax.set_ylabel("Sweeps / s")
    ax.set_title("Ising checkerboard Metropolis: CPU vs GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("benchmark_results.png", dpi=150)


if __name__ == "__main__":
    main()
