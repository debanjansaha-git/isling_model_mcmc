"""Generate SIMULATION_REPORT.md from the saved sweep and the benchmark report."""

import re
import time
from pathlib import Path

import numpy as np

from src.cpu_ising import T_C, CPUIsing2D
from src.data_io import load_simulation_hdf5


def _cpu_sweep_seconds(N: int, sweeps: int = 3) -> float:
    model = CPUIsing2D(N=N, T=T_C, init="random")
    t0 = time.perf_counter()
    model.metropolis_sweep(sweeps)
    return (time.perf_counter() - t0) / sweeps


def _benchmark_sections(path: str = "BENCHMARK_REPORT.md") -> str:
    """Pull the 'CPU vs GPU' tables out of the Phase 2 benchmark report."""
    p = Path(path)
    if not p.exists():
        return "_BENCHMARK_REPORT.md not found; run `uv run python -m src.benchmark`._\n"
    parts = re.split(r"^## ", p.read_text(), flags=re.M)
    keep = [s for s in parts if s.startswith("CPU vs GPU")]
    return "\n".join("### " + s.rstrip() + "\n" for s in keep)


def _fmt_time(s: float) -> str:
    return f"{s:.1f} s" if s < 600 else f"{s / 3600:.1f} h"


def write_report(data: dict | None = None, path: str = "SIMULATION_REPORT.md") -> None:
    data = data or load_simulation_hdf5("data/ising_sweep_N128.h5")
    md, sc = data["metadata"], data["scalars"]
    T, chi, cv = sc["T"], sc["susceptibility"], sc["specific_heat"]
    N, B = md["N"], md["batch"]
    total = md["therm_sweeps"] + md["meas_sweeps"]
    step = round(float(T[1] - T[0]), 4)

    t_chi, t_cv = float(T[np.argmax(chi)]), float(T[np.argmax(cv)])
    cpu_sweep = _cpu_sweep_seconds(N)
    cpu_total = cpu_sweep * total * B
    speedup = cpu_total / md["runtime_s"]

    def at(t: float, key: str) -> float:
        return float(sc[key][np.argmin(np.abs(T - t))])

    L = [
        "# 2D Ising Model — GPU Checkerboard MCMC: Simulation Report",
        "",
        "## Setup",
        "",
        f"- Lattice {N}×{N}, periodic boundaries, J = 1, h = 0, k_B = 1",
        f"- {B} temperatures, T ∈ [{T[0]:.2f}, {T[-1]:.2f}], step {step}; "
        "all chains run in one GPU batch",
        f"- {md['therm_sweeps']} thermalization + {md['meas_sweeps']} measurement sweeps "
        "(observables sampled after every measurement sweep)",
        "- Initial state: all +1 for T < T_c, random for T ≥ T_c (a random start at low T "
        "freezes into striped domains and never reaches the ordered equilibrium)",
        f"- Seed {md['seed']}; GPU: {md['gpu']}",
        "",
        "## Physical results",
        "",
        "| Quantity | Estimated T_peak | Onsager T_c | Offset |",
        "|---|---|---|---|",
        f"| Susceptibility χ | {t_chi:.2f} | {T_C:.4f} | {t_chi - T_C:+.3f} |",
        f"| Specific heat C_v | {t_cv:.2f} | {T_C:.4f} | {t_cv - T_C:+.3f} |",
        "",
        f"Peak positions are grid maxima (resolution {step}), single run, no error bars. "
        f"Peak heights: χ_max = {chi.max():.1f}, C_v,max = {cv.max():.2f}. Finite-size "
        "scaling predicts pseudo-critical points that sit slightly away from the "
        "infinite-system T_c at N=128, and autocorrelation times near T_c are long for "
        "single-spin Metropolis, so offsets of a few grid steps are expected and "
        "consistent with this result.",
        "",
        "| T | ⟨\\|M\\|⟩ | ⟨E⟩ |",
        "|---|---|---|",
    ]
    for t in (1.2, 2.26, 3.8):
        L.append(f"| {t} | {at(t, 'magnetization'):.4f} | {at(t, 'energy'):.4f} |")
    L += [
        "",
        "![phase transitions](phase_transitions.png)",
        "![domain snapshots](domain_snapshots.png)",
        "",
        "## Computational efficiency",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total sweeps per lattice | {total:,} ({md['therm_sweeps']:,} + {md['meas_sweeps']:,}) |",
        f"| Parallel lattices | {B} |",
        f"| Total spin updates | {B * N * N * total:.3e} |",
        f"| GPU wall time (incl. per-sweep measurements) | {md['runtime_s']:.1f} s |",
        f"| Throughput | {md['spin_updates_per_s']:.3e} spin-updates/s |",
        f"| Peak VRAM (torch allocator) | {md['peak_vram_mib']:.0f} MiB |",
        f"| CPU baseline, one {N}×{N} sweep (measured) | {cpu_sweep * 1e3:.0f} ms |",
        f"| CPU estimate for same workload (extrapolated) | {_fmt_time(cpu_total)} |",
        f"| Speedup vs CPU baseline | ≈ {speedup:,.0f}× |",
        "",
        "Memory throughput in bytes/s was not measured (no profiler counters were "
        "collected); spin-updates/s above is the measured proxy. VRAM is far below the "
        "1 GiB budget because 151 × 128² int8 spins occupy only 2.4 MiB.",
        "",
        "## Benchmark comparison: CPU single-spin vs GPU checkerboard",
        "",
        "From `BENCHMARK_REPORT.md` (CPU entries marked * are extrapolated).",
        "",
        _benchmark_sections(),
    ]
    Path(path).write_text("\n".join(L))
