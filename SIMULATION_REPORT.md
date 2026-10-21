# 2D Ising Model — GPU Checkerboard MCMC: Simulation Report

## Setup

- Lattice 128×128, periodic boundaries, J = 1, h = 0, k_B = 1
- 151 temperatures, T ∈ [1.00, 4.00], step 0.02; all chains run in one GPU batch
- 3000 thermalization + 10000 measurement sweeps (observables sampled after every measurement sweep)
- Initial state: all +1 for T < T_c, random for T ≥ T_c (a random start at low T freezes into striped domains and never reaches the ordered equilibrium)
- Seed 0; GPU: NVIDIA GeForce RTX 2060 SUPER

## Physical results

| Quantity | Estimated T_peak | Onsager T_c | Offset |
|---|---|---|---|
| Susceptibility χ | 2.28 | 2.2692 | +0.011 |
| Specific heat C_v | 2.26 | 2.2692 | -0.009 |

Peak positions are grid maxima (resolution 0.02), single run, no error bars. Peak heights: χ_max = 182.7, C_v,max = 2.15. Finite-size scaling predicts pseudo-critical points that sit slightly away from the infinite-system T_c at N=128, and autocorrelation times near T_c are long for single-spin Metropolis, so offsets of a few grid steps are expected and consistent with this result.

| T | ⟨\|M\|⟩ | ⟨E⟩ |
|---|---|---|
| 1.2 | 0.9970 | -1.9886 |
| 2.26 | 0.6403 | -1.4433 |
| 3.8 | 0.0137 | -0.5938 |

![phase transitions](phase_transitions.png)
![domain snapshots](domain_snapshots.png)

## Computational efficiency

| Metric | Value |
|---|---|
| Total sweeps per lattice | 13,000 (3,000 + 10,000) |
| Parallel lattices | 151 |
| Total spin updates | 3.216e+10 |
| GPU wall time (incl. per-sweep measurements) | 33.8 s |
| Throughput | 9.512e+08 spin-updates/s |
| Peak VRAM (torch allocator) | 43 MiB |
| CPU baseline, one 128×128 sweep (measured) | 51 ms |
| CPU estimate for same workload (extrapolated) | 27.6 h |
| Speedup vs CPU baseline | ≈ 2,936× |

Memory throughput in bytes/s was not measured (no profiler counters were collected); spin-updates/s above is the measured proxy. VRAM is far below the 1 GiB budget because 151 × 128² int8 spins occupy only 2.4 MiB.

## Benchmark comparison: CPU single-spin vs GPU checkerboard

From `BENCHMARK_REPORT.md` (CPU entries marked * are extrapolated).

### CPU vs GPU: 1000 sweeps, single lattice (B=1)

| N | CPU time | GPU time | Speedup | CPU flips/s | GPU flips/s |
|---|---|---|---|---|---|
| 32 | 3.67 s * | 0.48 s | 7.6x | 2.79e+05 | 2.12e+06 |
| 64 | 14.31 s * | 0.48 s | 29.9x | 2.86e+05 | 8.55e+06 |
| 128 | 57.24 s * | 0.48 s | 119.7x | 2.86e+05 | 3.43e+07 |
| 256 | 229.13 s * | 0.49 s | 472.0x | 2.86e+05 | 1.35e+08 |
| 512 | 0.26 h * | 0.49 s | 1,902.5x | 2.81e+05 | 5.34e+08 |

\* extrapolated from a timed run of fewer than 1000 sweeps.

### CPU vs GPU: best-case throughput (largest feasible batch)

| N | Best B | GPU flips/s | CPU flips/s | Speedup |
|---|---|---|---|---|
| 32 | 10000 | 1.29e+09 | 2.79e+05 | 4,627x |
| 64 | 10000 | 1.29e+09 | 2.86e+05 | 4,507x |
| 128 | 10000 | 1.28e+09 | 2.86e+05 | 4,486x |
| 256 | 10000 | 1.28e+09 | 2.86e+05 | 4,472x |
| 512 | 1000 | 1.27e+09 | 2.81e+05 | 4,530x |
