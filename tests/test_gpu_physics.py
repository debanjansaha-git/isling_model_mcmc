import numpy as np
import pytest
import torch

from src.cpu_ising import CPUIsing2D
from src.gpu_ising import GPUIsing2D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)
    torch.manual_seed(0)


def test_checkerboard_masks_partition_lattice() -> None:
    gpu = GPUIsing2D(N=6, T=1.0, batch=1)
    i, j = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    assert torch.equal(gpu.red.cpu(), (i + j) % 2 == 0)
    assert torch.equal(gpu.black.cpu(), (i + j) % 2 == 1)
    assert not (gpu.red & gpu.black).any()


@pytest.mark.parametrize("N", [4, 8, 17])
def test_energy_and_magnetization_match_cpu_exactly(N: int) -> None:
    cpu = CPUIsing2D(N=N, T=1.0, J=1.0, init="random")
    gpu = GPUIsing2D(N=N, T=1.0, batch=1, spins=cpu.spins[None])
    assert gpu.compute_energy()[0].item() == cpu.compute_energy()
    assert gpu.compute_magnetization()[0].item() == cpu.compute_magnetization()


def test_ground_state_energy_is_minus_two_per_spin() -> None:
    gpu = GPUIsing2D(N=4, T=1.0, batch=3, init="up")
    assert torch.all(gpu.compute_energy() == -2.0)


def test_sweep_preserves_spin_values_and_shape() -> None:
    gpu = GPUIsing2D(N=8, T=2.0, batch=5, init="random")
    gpu.sweep(10)
    assert gpu.spins.shape == (5, 8, 8)
    assert set(gpu.spins.unique().tolist()) <= {-1, 1}


def test_per_lattice_temperatures_are_independent() -> None:
    T = torch.tensor([0.1] * 4 + [10.0] * 4)
    gpu = GPUIsing2D(N=16, T=T, batch=8, init="up")
    gpu.sweep(100)
    M = gpu.compute_magnetization()
    assert M[:4].min() > 0.95
    assert M[4:].max() < 0.35


def test_chunked_sweep_matches_semantics() -> None:
    gpu = GPUIsing2D(N=16, T=0.1, batch=8, init="up", max_chunk_spins=16 * 16 * 3)
    gpu.sweep(50)
    assert gpu.compute_magnetization().min() > 0.95


@pytest.mark.parametrize("T, tol_m, tol_e", [(1.5, 0.03, 0.03), (3.5, 0.05, 0.05)])
def test_equilibrium_matches_cpu(T: float, tol_m: float, tol_e: float) -> None:
    N, burn, measure = 16, 200, 1000

    cpu = CPUIsing2D(N=N, T=T, init="up")
    cpu.metropolis_sweep(burn)
    cm, ce = [], []
    for _ in range(measure):
        cpu.metropolis_sweep(1)
        cm.append(cpu.compute_magnetization())
        ce.append(cpu.compute_energy())

    gpu = GPUIsing2D(N=N, T=T, batch=64, init="up")
    gpu.sweep(burn)
    gm, ge = [], []
    for _ in range(measure):
        gpu.sweep(1)
        gm.append(gpu.compute_magnetization().mean().item())
        ge.append(gpu.compute_energy().mean().item())

    assert np.mean(gm) == pytest.approx(np.mean(cm), abs=tol_m)
    assert np.mean(ge) == pytest.approx(np.mean(ce), abs=tol_e)
