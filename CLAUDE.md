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
- Run tests: `uv run pytest`

### Code Style Guidelines
- Use Type Hints (`torch.Tensor`, `np.ndarray`, `float`, `int`) for public API functions.
- Avoid Python loops over lattice grid points; use vectorized slice/shift operations (`torch.roll`).
- Store lattice states as integer tensors (`torch.int8` or `torch.int16`) or float32 when required by probability computations.
- Ensure reproducible random seeds across tests using `torch.manual_seed()` and `numpy.random.seed()`.

## Milestone Roadmap
- [x] Environment and `CLAUDE.md` initialization.
- [ ] **Phase 1 (Days 1–3)**: CPU single-spin baseline implementation & unit testing. Validate $\vert{}M\vert{}$ drop at $T_c \approx 2.269$.
- [ ] **Phase 2 (Days 4–7)**: Vectorized PyTorch Checkerboard GPU engine. Benchmark speedup vs CPU.
- [ ] **Phase 3 (Days 8–10)**: Domain visualizer & HDF5 dataset exporter across temperature sweeps $T \in [1.0, 4.0]$.