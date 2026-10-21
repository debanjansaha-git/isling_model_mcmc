# Benchmark Report

- GPU: NVIDIA GeForce RTX 2060 SUPER, torch 2.5.1+cu121, CUDA 12.1
- CPU: pure-Python single-spin Metropolis (`CPUIsing2D`), one lattice
- GPU: checkerboard engine, int8 spins, per-lattice temperatures T in [1, 4]
- One *sweep* = N² spin-update proposals per lattice. Sweeps/s below are batch sweeps (all B lattices advance together).
- Configurations whose int8 lattice storage exceeds 1 GiB are skipped (`—`).

## GPU throughput: sweeps/s

| N \ B | 1 | 100 | 1000 | 10000 |
|---|---|---|---|---|
| 32 | 2,097.8 | 2,098.7 | 1,200.1 | 126.2 |
| 64 | 2,132.5 | 1,950.3 | 305.4 | 31.5 |
| 128 | 2,073.0 | 773.2 | 78.4 | 7.8 |
| 256 | 2,095.2 | 191.7 | 19.5 | 2.0 |
| 512 | 2,049.1 | 48.7 | 4.9 | — |

## GPU throughput: spin-flips/s

| N \ B | 1 | 100 | 1000 | 10000 |
|---|---|---|---|---|
| 32 | 2.15e+06 | 2.15e+08 | 1.23e+09 | 1.29e+09 |
| 64 | 8.73e+06 | 7.99e+08 | 1.25e+09 | 1.29e+09 |
| 128 | 3.40e+07 | 1.27e+09 | 1.28e+09 | 1.28e+09 |
| 256 | 1.37e+08 | 1.26e+09 | 1.28e+09 | 1.28e+09 |
| 512 | 5.37e+08 | 1.28e+09 | 1.27e+09 | — |

## CPU vs GPU: 1000 sweeps, single lattice (B=1)

| N | CPU time | GPU time | Speedup | CPU flips/s | GPU flips/s |
|---|---|---|---|---|---|
| 32 | 3.67 s * | 0.48 s | 7.6x | 2.79e+05 | 2.12e+06 |
| 64 | 14.31 s * | 0.48 s | 29.9x | 2.86e+05 | 8.55e+06 |
| 128 | 57.24 s * | 0.48 s | 119.7x | 2.86e+05 | 3.43e+07 |
| 256 | 229.13 s * | 0.49 s | 472.0x | 2.86e+05 | 1.35e+08 |
| 512 | 0.26 h * | 0.49 s | 1,902.5x | 2.81e+05 | 5.34e+08 |

\* extrapolated from a timed run of fewer than 1000 sweeps.

## CPU vs GPU: best-case throughput (largest feasible batch)

| N | Best B | GPU flips/s | CPU flips/s | Speedup |
|---|---|---|---|---|
| 32 | 10000 | 1.29e+09 | 2.79e+05 | 4,627x |
| 64 | 10000 | 1.29e+09 | 2.86e+05 | 4,507x |
| 128 | 10000 | 1.28e+09 | 2.86e+05 | 4,486x |
| 256 | 10000 | 1.28e+09 | 2.86e+05 | 4,472x |
| 512 | 1000 | 1.27e+09 | 2.81e+05 | 4,530x |
