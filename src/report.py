"""Generate SIMULATION_REPORT.md and README.md from the saved sweep and FSS data."""

import functools
import re
import time
from pathlib import Path

import numpy as np

from src.cpu_ising import T_C, CPUIsing2D
from src.data_io import load_fss_hdf5, load_simulation_hdf5
from src.fss import find_crossing

SWEEP_PATH = "data/ising_sweep_N128.h5"
FSS_PATH = "data/fss_analysis.h5"
GAMMA, NU = 1.75, 1.0


@functools.cache
def _cpu_sweep_seconds(N: int, sweeps: int = 3) -> float:
    model = CPUIsing2D(N=N, T=T_C, init="random")
    t0 = time.perf_counter()
    model.metropolis_sweep(sweeps)
    return (time.perf_counter() - t0) / sweeps


def _fmt_time(s: float) -> str:
    return f"{s:.1f} s" if s < 600 else f"{s / 3600:.1f} h"


def _benchmark_sections(path: str = "BENCHMARK_REPORT.md") -> str:
    """Pull the 'CPU vs GPU' tables out of the Phase 2 benchmark report."""
    p = Path(path)
    if not p.exists():
        return "_BENCHMARK_REPORT.md not found; run `uv run python -m src.benchmark`._\n"
    parts = re.split(r"^## ", p.read_text(), flags=re.M)
    keep = [s for s in parts if s.startswith("CPU vs GPU")]
    return "\n".join("### " + s.rstrip() + "\n" for s in keep)


def _benchmark_best_case(path: str = "BENCHMARK_REPORT.md") -> dict | None:
    """Parse the best-case rows: {'gpu_flips': max, 'speedup_min', 'speedup_max'}."""
    p = Path(path)
    if not p.exists():
        return None
    rows = re.findall(
        r"^\| \d+ \| \d+ \| ([\d.e+]+) \| ([\d.e+]+) \| ([\d,]+)x \|$", p.read_text(), re.M
    )
    if not rows:
        return None
    sp = [int(r[2].replace(",", "")) for r in rows]
    return {"gpu_flips": max(float(r[0]) for r in rows), "speedup_min": min(sp),
            "speedup_max": max(sp)}


def _measure_vram() -> list[tuple[str, int, int, float, float]]:
    """(label, N, B, int8 lattice MiB, measured peak MiB) for representative workloads."""
    import torch

    from src.gpu_ising import GPUIsing2D

    if not torch.cuda.is_available():
        return []
    configs = [("FSS run (one size)", 128, 51), ("Production sweep", 128, 151),
               ("Large batch", 512, 1000), ("Max batch (CLI default)", 128, 10000)]
    out = []
    for label, N, B in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = GPUIsing2D(N=N, T=torch.linspace(1.0, 4.0, B), batch=B, init="up")
        model.sweep(1)
        torch.cuda.synchronize()
        out.append((label, N, B, B * N * N / 2**20, torch.cuda.max_memory_allocated() / 2**20))
        del model
    return out


def _at(r: dict, key: str, t: float) -> float:
    return float(r[key][np.argmin(np.abs(r["T"] - t))])


def _fss_summary(fss: dict) -> dict:
    res = fss["results"]
    sizes = sorted(res)
    T_eval = float(res[sizes[0]]["T"][np.argmin(np.abs(res[sizes[0]]["T"] - 2.27))])
    crossings = []
    for a, b in zip(sizes, sizes[1:]):
        Tx = find_crossing(res[a]["T"], res[a]["binder"], res[b]["binder"])
        crossings.append((a, b, Tx))
    tau_c = np.array([_at(res[N], "tau_int_M", T_eval) for N in sizes])
    tau_max = np.array([res[N]["tau_int_M"].max() for N in sizes])
    logN = np.log(sizes)
    return {
        "sizes": sizes, "res": res, "T_eval": T_eval, "crossings": crossings,
        "tau_c": tau_c, "tau_max": tau_max,
        "z_c": float(np.polyfit(logN, np.log(tau_c), 1)[0]),
        "z_max": float(np.polyfit(logN, np.log(tau_max), 1)[0]),
    }


def _fss_markdown(fss: dict, meas: int, n_blocks: int) -> list[str]:
    s = _fss_summary(fss)
    res, sizes, Te = s["res"], s["sizes"], s["T_eval"]
    L = ["## Finite-size scaling and error analysis", "",
         f"Dataset: N ∈ {{{', '.join(map(str, sizes))}}}, T ∈ [{res[sizes[0]]['T'][0]:.3f}, "
         f"{res[sizes[0]]['T'][-1]:.3f}] in steps of 0.005 ({len(res[sizes[0]]['T'])} "
         f"temperatures), {fss['metadata']['therm_sweeps']:,} thermalization + {meas:,} "
         "measurement sweeps per lattice, observables sampled after every sweep. Errors "
         f"are blocked-jackknife with {n_blocks} blocks of {meas // n_blocks:,} sweeps.", "",
         "### Binder cumulants U₄", "",
         "U₄ = 1 − ⟨M⁴⟩ / (3⟨M²⟩²) tends to 2/3 in the ordered phase and to 0 for "
         "Gaussian (disordered) fluctuations. Curves for different N intersect near T_c "
         "because U₄ is a universal function of (T − T_c)N^{1/ν} up to corrections to scaling.",
         "", "| Pair | Crossing T | Offset from Onsager T_c |", "|---|---|---|"]
    for a, b, Tx in s["crossings"]:
        L.append(f"| N={a} / N={b} | " + (f"{Tx:.4f} | {Tx - T_C:+.4f} |" if Tx else "none | — |"))
    L += ["", f"Onsager T_c = {T_C:.4f}. Both crossings lie at or slightly below T_c, "
          "spanning "
          f"{min(c[2] for c in s['crossings']):.3f}–{max(c[2] for c in s['crossings']):.3f}. "
          "The larger-N pair is noisier: near T_c the N=128 chains contain only a few dozen "
          "independent samples (see below), and the crossing is a linear interpolation "
          "between 0.005-spaced points whose U₄ error bars overlap. Scatter of this size "
          "(≈ 0.01 in T) is therefore expected and no systematic shift is claimed.",
          "", "![Binder cumulants](binder_cumulants.png)", "",
          "### Critical slowing down", "",
          "Integrated autocorrelation time τ_int of |M| (Sokal automatic window, "
          "W ≥ 6 τ_int(W)), in sweeps:", "",
          f"| N | τ_int at T={Te:.2f} | max over T scan (at T) |", "|---|---|---|"]
    for N, tc, tm in zip(sizes, s["tau_c"], s["tau_max"]):
        Tm = float(res[N]["T"][np.argmax(res[N]["tau_int_M"])])
        L.append(f"| {N} | {tc:,.0f} | {tm:,.0f} (T={Tm:.3f}) |")
    L += ["",
          f"The maximum grows from {s['tau_max'][0]:,.0f} sweeps (N={sizes[0]}) to "
          f"{s['tau_max'][-1]:,.0f} sweeps (N={sizes[-1]}). Fitting τ ∝ N^z gives "
          f"z = {s['z_c']:.2f} using τ at T={Te:.2f} (literature value for single-spin "
          f"Metropolis ≈ 2.17) and z = {s['z_max']:.2f} using the scan maximum. The "
          "maximum is a noisy statistic (it picks the largest of 51 estimates at each N), "
          "so the fixed-temperature fit is the better estimate of z.", "",
          "Why local updates degrade: single-spin Metropolis moves one spin at a time, so "
          "the correlated clusters of size ξ ~ N that appear at T_c relax only by diffusing "
          "their boundaries. The relaxation time therefore scales as ξ^z with z ≈ 2.17, "
          f"i.e. a {2 ** s['z_c']:.1f}× cost for each doubling of N, on top of the 4× cost "
          "of the larger lattice. Cluster algorithms (Wolff, Swendsen–Wang) have z ≈ 0.25 "
          "and would remove most of this penalty; they are not implemented here.", "",
          "### Statistical errors (blocked jackknife)", "",
          f"Values at T={Te:.2f} (nearest grid point to T_c), mean ± jackknife error:", "",
          "| N | U₄ | χ | C_v | n_eff ≈ meas / (2 τ_int) |", "|---|---|---|---|---|"]
    for N, tc in zip(sizes, s["tau_c"]):
        r = res[N]
        L.append(f"| {N} | {_at(r, 'binder', Te):.4f} ± {_at(r, 'binder_err', Te):.4f} | "
                 f"{_at(r, 'susceptibility', Te):.1f} ± {_at(r, 'susceptibility_err', Te):.1f} | "
                 f"{_at(r, 'specific_heat', Te):.3f} ± {_at(r, 'specific_heat_err', Te):.3f} | "
                 f"{meas / (2 * tc):.0f} |")
    L += ["",
          "Errors grow with N at fixed sweep count because τ_int grows faster than the "
          "number of samples. Caveat: the jackknife block length is "
          f"{meas // n_blocks:,} sweeps, which exceeds τ_int at T_c for all sizes "
          f"(N=128: {s['tau_c'][-1]:,.0f}) but is shorter than the N=128 scan maximum "
          f"({s['tau_max'][-1]:,.0f}); error bars for N=128 in the immediate vicinity of "
          "the τ_int peak may be underestimated.", "",
          "### Universal data collapse", "",
          f"Rescaling χ → χ N^(−γ/ν) and T → (T − T_c) N^(1/ν) with the exact 2D Ising "
          f"exponents γ = {GAMMA}, ν = {NU} and the exact T_c collapses all sizes onto one "
          "curve:", "", "![Susceptibility collapse](fss_data_collapse.png)", "",
          "| N | χ_max N^(−γ/ν) | T at χ_max |", "|---|---|---|"]
    for N in sizes:
        r = res[N]
        k = int(np.argmax(r["susceptibility"]))
        L.append(f"| {N} | {r['susceptibility'][k] * N ** (-GAMMA / NU):.4f} | {r['T'][k]:.3f} |")
    L += ["", "The scaled peak height is constant to within a few percent across sizes "
          "(N=32 is slightly lower, as expected from corrections to scaling). The collapse "
          "is a visual and peak-height check, not a fitted exponent estimate.", ""]
    return L


def write_report(data: dict | None = None, path: str = "SIMULATION_REPORT.md") -> None:
    data = data or load_simulation_hdf5(SWEEP_PATH)
    fss = load_fss_hdf5(FSS_PATH)
    md, sc = data["metadata"], data["scalars"]
    T, chi, cv = sc["T"], sc["susceptibility"], sc["specific_heat"]
    N, B = md["N"], md["batch"]
    total = md["therm_sweeps"] + md["meas_sweeps"]
    step = round(float(T[1] - T[0]), 4)

    t_chi, t_cv = float(T[np.argmax(chi)]), float(T[np.argmax(cv)])
    cpu_sweep = _cpu_sweep_seconds(N)
    cpu_total = cpu_sweep * total * B
    speedup = cpu_total / md["runtime_s"]
    best = _benchmark_best_case()

    def at(t: float, key: str) -> float:
        return float(sc[key][np.argmin(np.abs(T - t))])

    L = [
        "# 2D Ising Model — GPU Checkerboard MCMC: Simulation Report",
        "",
        "## Setup",
        "",
        f"- Square lattice, periodic boundaries, J = 1, h = 0, k_B = 1; Metropolis dynamics "
        "with Red/Black checkerboard decomposition (GPU) or sequential single-spin (CPU)",
        f"- Production sweep: {N}×{N}, {B} temperatures T ∈ [{T[0]:.2f}, {T[-1]:.2f}], "
        f"step {step}, all chains in one GPU batch; {md['therm_sweeps']:,} thermalization + "
        f"{md['meas_sweeps']:,} measurement sweeps, observables sampled every sweep",
        "- Initial state: all +1 for T < T_c, random for T ≥ T_c (a random start at low T "
        "freezes into striped domains and never reaches the ordered equilibrium)",
        f"- Seed {md['seed']}; GPU: {md['gpu']}",
        "",
        "## Physical results (N=128 temperature sweep)",
        "",
        "| Quantity | Estimated T_peak | Onsager T_c | Offset |",
        "|---|---|---|---|",
        f"| Susceptibility χ | {t_chi:.2f} | {T_C:.4f} | {t_chi - T_C:+.3f} |",
        f"| Specific heat C_v | {t_cv:.2f} | {T_C:.4f} | {t_cv - T_C:+.3f} |",
        "",
        f"Peak positions are grid maxima (resolution {step}) from a single run. Peak "
        f"heights: χ_max = {chi.max():.1f}, C_v,max = {cv.max():.2f}. Finite-size effects "
        "shift pseudo-critical points slightly from the infinite-system T_c and "
        "autocorrelation times near T_c are long (see below), so offsets of a grid step "
        "or two are expected.",
        "",
        "| T | ⟨\\|M\\|⟩ | ⟨E⟩ |",
        "|---|---|---|",
    ]
    for t in (1.2, 2.26, 3.8):
        L.append(f"| {t} | {at(t, 'magnetization'):.4f} | {at(t, 'energy'):.4f} |")
    L += ["", "![phase transitions](phase_transitions.png)", "",
          "![domain snapshots](domain_snapshots.png)", ""]

    L += _fss_markdown(fss, fss["metadata"]["meas_sweeps"], fss["metadata"]["n_blocks"])

    L += [
        "## Hardware & execution summary",
        "",
        f"Hardware: {md['gpu']} (8 GB, compute capability 7.5), torch with CUDA 12.1.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Production sweep: sweeps × lattices | {total:,} × {B} |",
        f"| Total spin updates | {B * N * N * total:.3e} |",
        f"| GPU wall time (incl. per-sweep measurements) | {md['runtime_s']:.1f} s |",
        f"| Throughput during production run | {md['spin_updates_per_s']:.3e} spin-updates/s |",
    ]
    if best:
        L.append(f"| Peak raw engine throughput (benchmark, large batch) | "
                 f"{best['gpu_flips']:.2e} spin-updates/s |")
    L += [
        f"| CPU baseline, one {N}×{N} sweep (measured) | {cpu_sweep * 1e3:.0f} ms |",
        f"| CPU estimate for the production workload (extrapolated) | {_fmt_time(cpu_total)} |",
        f"| Speedup on the production workload | ≈ {speedup:,.0f}× |",
    ]
    if best:
        L.append(f"| Speedup, raw engine throughput (best-case batches) | "
                 f"{best['speedup_min']:,}–{best['speedup_max']:,}× |")
    L += ["",
          "The production run is slower than the raw engine rate because it measures M "
          "and E after every sweep (extra kernels per sweep). Speedups compare against "
          "the pure-Python CPU baseline and are extrapolated from short CPU timings; "
          "they do not compare against an optimized (e.g. C/numba, multispin-coded) CPU "
          "implementation.", "",
          "### VRAM usage (peak torch allocation, one sweep)", "",
          "| Workload | N | B | int8 lattices | Peak VRAM |", "|---|---|---|---|---|"]
    for label, n, b, lat, peak in _measure_vram():
        L.append(f"| {label} | {n} | {b:,} | {lat:.1f} MiB | {peak:.0f} MiB |")
    L += ["", "Float32 temporaries (random numbers, ΔE, Boltzmann factors) are built per "
          "chunk of at most 2²⁵ spins, so peak VRAM stays well under the 1 GiB budget "
          "even for the 10,000 × 128² batch. Memory traffic in bytes/s was not profiled.",
          "", "## Benchmark comparison: CPU single-spin vs GPU checkerboard", "",
          "From `BENCHMARK_REPORT.md` (CPU entries marked * are extrapolated; see "
          "that file for the full N × B throughput grid and `benchmark_results.png`).", "",
          _benchmark_sections(), "",
          "## Limitations", "",
          "- Single seed per dataset; peak locations are grid maxima without error bars.",
          "- Local updates suffer critical slowing down; N=128 near T_c has few "
          "independent samples and the jackknife may be optimistic there (see above).",
          "- The data collapse uses the exact exponents rather than fitting them.",
          "- χ is defined with ⟨|M|⟩² (the standard finite-size convention for h = 0).",
          ""]
    Path(path).write_text("\n".join(L))


def write_readme(data: dict | None = None, path: str = "README.md") -> None:
    data = data or load_simulation_hdf5(SWEEP_PATH)
    fss = load_fss_hdf5(FSS_PATH)
    md, sc = data["metadata"], data["scalars"]
    T, chi, cv = sc["T"], sc["susceptibility"], sc["specific_heat"]
    s = _fss_summary(fss)
    best = _benchmark_best_case()
    cpu_sweep = _cpu_sweep_seconds(md["N"])
    cpu_total = cpu_sweep * (md["therm_sweeps"] + md["meas_sweeps"]) * md["batch"]
    speedup = cpu_total / md["runtime_s"]
    t_chi, t_cv = float(T[np.argmax(chi)]), float(T[np.argmax(cv)])
    xs = [c for c in s["crossings"] if c[2] is not None]
    xs_txt = ", ".join(f"N={a}/{b}: {Tx:.3f}" for a, b, Tx in xs)

    gpu_rate = (f"{best['gpu_flips']:.1e}" if best else f"{md['spin_updates_per_s']:.1e}")
    speed_rows = []
    if best:
        speed_rows.append(f"| Raw engine, large batches | {best['speedup_min']:,}–"
                          f"{best['speedup_max']:,}× |")
    speed_rows.append(f"| Full production run (13,000 sweeps × {md['batch']} lattices, "
                      f"with per-sweep measurement) | ≈ {speedup:,.0f}× |")

    R = f"""# GPU-Accelerated 2D Ising Model — Monte Carlo with Error Analysis

## Executive summary

Metropolis Markov chain Monte Carlo for the 2D Ising model on an NVIDIA RTX 2060 Super.
A checkerboard (Red/Black) decomposition updates half the lattice in parallel with
PyTorch tensor operations, running hundreds to thousands of independent lattices, each
at its own temperature, in one batch. The study goes beyond a phase-transition sweep:
Binder cumulants locate T_c from finite-size crossings, integrated autocorrelation times
quantify critical slowing down, and blocked-jackknife resampling gives error bars on the
non-linear observables χ, C_v and U₄. A sequential CPU implementation serves as the
correctness reference and the performance baseline.

## Key results at a glance

**Throughput** (pure-Python single-spin CPU baseline vs GPU):

| | CPU baseline | GPU checkerboard |
|---|---|---|
| Spin updates / s | ≈ {md['N'] ** 2 / cpu_sweep:.1e} | up to ≈ {gpu_rate} |

| Workload | GPU speedup |
|---|---|
{chr(10).join(speed_rows)}

The baseline is unoptimized Python; the speedup is not a comparison with tuned CPU code.
CPU times are extrapolated from short runs.

**Physics** (N = {md['N']} sweep, T step 0.02; FSS, N ∈ {{{', '.join(map(str, s['sizes']))}}}):

| Observable | Estimate | Exact (Onsager) |
|---|---|---|
| χ peak position | T = {t_chi:.2f} | T_c = {T_C:.4f} |
| C_v peak position | T = {t_cv:.2f} | T_c = {T_C:.4f} |
| Binder-cumulant crossings | {xs_txt} | T_c = {T_C:.4f} |
| Autocorrelation exponent z (τ ∝ N^z) | {s['z_c']:.2f} | ≈ 2.17 (Metropolis, literature) |
| χ_max N^(−7/4), N = {s['sizes'][0]}/{s['sizes'][1]}/{s['sizes'][2]} | {" / ".join(f"{float(s['res'][N]['susceptibility'].max()) * N ** -1.75:.3f}" for N in s['sizes'])} | constant (scaling) |

Critical slowing down: τ_int(|M|) at T = {s['T_eval']:.2f} grows from {s['tau_c'][0]:,.0f}
to {s['tau_c'][-1]:,.0f} sweeps between N = {s['sizes'][0]} and N = {s['sizes'][-1]}
(scan maximum: {s['tau_max'][0]:,.0f} → {s['tau_max'][-1]:,.0f}).

## Visualizations

![Phase transition observables](phase_transitions.png)

![Domain snapshots at T = 1.2, 2.26, 3.8](domain_snapshots.png)

![Binder cumulant crossings](binder_cumulants.png)

![Susceptibility data collapse](fss_data_collapse.png)

![Benchmark: CPU vs GPU](benchmark_results.png)

## Physics & methodology

**Hamiltonian.** H(σ) = −J Σ⟨i,j⟩ σᵢσⱼ − h Σᵢ σᵢ, σᵢ ∈ {{−1, +1}}, J = 1, h = 0, k_B = 1,
periodic boundaries. Onsager: T_c = 2J / ln(1 + √2) ≈ {T_C:.6f}.

**Observables** (per spin, M = Σσ / N²): χ = (N²/T)(⟨M²⟩ − ⟨|M|⟩²),
C_v = (N²/T²)(⟨E²⟩ − ⟨E⟩²), Binder cumulant U₄ = 1 − ⟨M⁴⟩ / (3⟨M²⟩²)
(→ 2/3 ordered, → 0 for Gaussian fluctuations).

**Checkerboard decomposition.** Sites with (i + j) mod 2 = 0 (Red) have only Black
neighbors, so all Red spins can be updated simultaneously without violating detailed
balance; then all Black spins are updated using the new Red values. A flip is accepted
with probability min(1, e^(−ΔE/T)), ΔE = 2σ(J Σ_nn σ + h); neighbor sums use `torch.roll`.
Each lattice carries its own temperature in a (B, 1, 1) tensor. Spins are int8.

**Autocorrelation (Sokal window).** C(t) is the normalized autocovariance (FFT). The
integrated time is τ_int(W) = ½ + Σ_{{t=1}}^{{W}} C(t), with the window W chosen as the
smallest W satisfying W ≥ 6 τ_int(W). The variance of a sample mean is 2τ_int σ² / n.

**Blocked jackknife.** Split the series into n blocks, compute the statistic θ₍ᵢ₎ with
block i removed. Bias-corrected estimate θ̂ = nθ − (n−1)θ̄₍·₎, error
σ² = (n−1)/n Σᵢ (θ₍ᵢ₎ − θ̄₍·₎)². Blocks longer than τ_int make the error account for
correlations; this works for non-linear statistics such as χ, C_v and U₄.

## Quickstart

Requires Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with CUDA 12.1
drivers (the CPU baseline runs without a GPU).

```bash
uv sync                                    # install dependencies (CUDA 12.1 torch wheel)
uv run pytest                              # full test suite
uv run python -m src.cpu_ising             # CPU demo
uv run python -m src.gpu_ising --batch 10000 --lattice 128   # GPU throughput
uv run python -m src.benchmark             # CPU vs GPU benchmark -> BENCHMARK_REPORT.md
uv run python -m src.simulate              # N=128 sweep -> data/ising_sweep_N128.h5
uv run python -m src.fss                   # FSS dataset -> data/fss_analysis.h5 (~5 min)
uv run python -m src.visualize             # sweep figures
uv run python -m src.visualize_fss         # Binder + collapse figures
uv run python -m src.report                # regenerate SIMULATION_REPORT.md and README.md
```

`data/*.h5` files are git-ignored; regenerate them with the commands above.

## Project layout

| Path | Purpose |
|---|---|
| `src/cpu_ising.py` | Sequential single-spin Metropolis baseline |
| `src/gpu_ising.py` | Batched checkerboard engine (PyTorch CUDA) |
| `src/simulate.py` | Production temperature sweep |
| `src/stats.py` | Autocorrelation and jackknife |
| `src/fss.py` | Binder cumulant and finite-size-scaling runs |
| `src/data_io.py` | HDF5 read/write |
| `src/visualize.py`, `src/visualize_fss.py` | Figures |
| `src/benchmark.py`, `src/report.py` | Benchmark and report generation |

Full details, tables and caveats: [SIMULATION_REPORT.md](SIMULATION_REPORT.md),
[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).
"""
    Path(path).write_text(R)


def main() -> None:
    data = load_simulation_hdf5(SWEEP_PATH)
    write_report(data)
    write_readme(data)
    print("wrote SIMULATION_REPORT.md and README.md")


if __name__ == "__main__":
    main()
