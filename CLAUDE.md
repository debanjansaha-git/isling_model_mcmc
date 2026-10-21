# CLAUDE.md - GPU-Accelerated 2D Ising Model via MCMC

## Project Overview
A high-performance Monte Carlo Markov Chain (MCMC) simulation of the 2D Ising model on a square lattice. Utilizes PyTorch CUDA tensor vectorization and Checkerboard (Red-Black) sub-lattice decomposition to simulate tens of thousands of parallel lattices across temperature sweeps.

## Core Physics & Theoretical Parameters
- **Hamiltonian**: $H(\sigma) = -J \sum_{\langle i, j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i$
- **Spin Values**: $\sigma_i \in \{-1, +1\}$
- **Coupling Constant ($J$)**: $J = 1.0$ (Ferromagnetic interaction)
- **External Field ($h$)**: $h = 0.0$ (default baseline)
- **Boltzmann Constant ($k_B$)**: $k_B = 1.0$ (dimensionless units)
- **Critical Temperature ($T_c$)**: $T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269185$
- **Observed Quantities**:
  - Magnetization per spin: $\vert{}M\vert{} = \frac{1}{N^2} \left\vert{} \sum_i \sigma_i \right\vert{}$
  - Energy per spin: $E = \frac{H(\sigma)}{N^2}$
  - Magnetic Susceptibility: $\chi = \frac{N^2}{T} (\langle M^2 \rangle - \langle \vert{}M\vert{} \rangle^2)$
  - Specific Heat: $C_v = \frac{N^2}{T^2} (\langle E^2 \rangle - \langle E \rangle^2)$

## Architecture & Algorithm Design
1. **CPU Baseline (`src/cpu_ising.py`)**:
   - Single-spin sequential Metropolis-Hastings update.
   - Periodic boundary conditions via numpy roll or modular indexing.
   - Ground truth verification against theoretical phase transition at $T_c$.

2. **GPU Vectorized Engine (`src/gpu_ising.py`)**:
   - **Checkerboard Sub-lattice Decomposition**:
     - Red sub-lattice: $(i + j) \% 2 == 0$
     - Black sub-lattice: $(i + j) \% 2 == 1$
   - Updates applied simultaneously to all Red sites (using Black site fixed neighbors), followed by Black sites (using updated Red site neighbors).
   - Tensor shape: Batch size $B \times N \times N$, processing $B$ parallel independent Markov chains across different temperatures $T \in [1.0, 4.0]$.

3. **Data Pipeline & Storage (`src/data_io.py`)**:
   - Export equilibrium configurations and scalar ensembles to HDF5 (`.h5`).

4. **Statistics & Finite-Size Scaling (Research Expansion)**:
   - `src/stats.py`: `compute_autocorrelation(series)` returns normalized $C(t)$ and $\tau_{int} = \tfrac12 + \sum_{t=1}^{W} C(t)$ (Sokal automatic window; white noise gives $\tfrac12$). `jackknife_resampling(series, func, n_blocks)` returns a bias-corrected estimate and error; blocks must exceed $\tau_{int}$ for correlated MCMC data.
   - `src/fss.py`: Binder cumulant $U_4 = 1 - \frac{\langle M^4 \rangle}{3 \langle M^2 \rangle^2}$ ($\to 2/3$ ordered, $\to 0$ for Gaussian/disordered $M$; universal crossing near $T_c$). Runs $N \in \{32, 64, 128\}$, $T \in [2.15, 2.40]$, step 0.005, keeping per-sweep $M$/$E$ series, and writes `data/fss_analysis.h5` (groups `N_<size>`).
   - `src/visualize_fss.py`: `binder_cumulants.png` and `fss_data_collapse.png` ($\chi N^{-\gamma/\nu}$ vs $(T-T_c)N^{1/\nu}$, $\gamma = 7/4$, $\nu = 1$).
   - Chains start all $+1$ for $T < T_c$ and random above (random low-$T$ starts freeze into stripe domains). Metropolis critical slowing down ($\tau_{int}$ grows as $N^{z}$, $z \approx 2.17$) sets the required sweep count.

## Tech Stack & Hardware Target
- **Language**: Python 3.11+
- **Environment Manager**: `uv`
- **Core Libraries**: `torch` (CUDA 12.1+), `numpy`, `h5py`, `matplotlib`, `pytest`
- **Hardware Target**: NVIDIA RTX 2060 Super (8 GB VRAM, Compute Capability 7.5)
- **VRAM Constraint**: Keep batched lattice allocations $< 1.0\text{ GB}$ VRAM.

## Common Commands

### Development & Execution
- Run CPU baseline test: `uv run python -m src.cpu_ising`
- Run GPU benchmark: `uv run python -m src.gpu_ising --batch 10000 --lattice 128`
- Generate phase transition plots: `uv run python -m src.visualize`
- Run production temperature sweep (N=128): `uv run python -m src.simulate`
- Run finite-size scaling dataset: `uv run python -m src.fss`
- Generate FSS plots: `uv run python -m src.visualize_fss`
- Run tests: `uv run pytest`

### Code Style Guidelines
- Use Type Hints (`torch.Tensor`, `np.ndarray`, `float`, `int`) for public API functions.
- Avoid Python loops over lattice grid points; use vectorized slice/shift operations (`torch.roll`).
- Store lattice states as integer tensors (`torch.int8` or `torch.int16`) or float32 when required by probability computations.
- Ensure reproducible random seeds across tests using `torch.manual_seed()` and `numpy.random.seed()`.

## Milestone Roadmap
- [x] Environment and `CLAUDE.md` initialization.
- [x] **Phase 1 (Days 1–3)**: CPU single-spin baseline implementation & unit testing. Validate $\vert{}M\vert{}$ drop at $T_c \approx 2.269$.
- [x] **Phase 2 (Days 4–7)**: Vectorized PyTorch Checkerboard GPU engine. Benchmark speedup vs CPU.
- [x] **Phase 3 (Days 8–10)**: Domain visualizer & HDF5 dataset exporter across temperature sweeps $T \in [1.0, 4.0]$.
- [ ] **Research Phase 1**: Autocorrelation analysis, jackknife error bars, Binder cumulant crossings and $\chi$ data collapse for $N \in \{32, 64, 128\}$.