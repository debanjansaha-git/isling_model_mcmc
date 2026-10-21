import numpy as np
import pytest

from src.cpu_ising import CPUIsing2D


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)


def test_ground_state_energy_4x4_is_minus_two_per_spin() -> None:
    model = CPUIsing2D(N=4, T=1.0, init="up")
    assert model.compute_energy() == pytest.approx(-2.0)


def test_delta_e_matches_neighbor_formula_and_brute_force() -> None:
    model = CPUIsing2D(N=6, T=1.0, init="random")
    N = model.N
    for i in range(N):
        for j in range(N):
            s = model.spins[i, j]
            nb = (
                model.spins[(i + 1) % N, j]
                + model.spins[(i - 1) % N, j]
                + model.spins[i, (j + 1) % N]
                + model.spins[i, (j - 1) % N]
            )
            dE = model.delta_energy(i, j)
            # H = -J sum s_i s_j, so flipping s changes H by +2*J*s*sum(nb)
            assert dE == pytest.approx(2 * model.J * s * nb)

            before = model.compute_energy() * N * N
            model.spins[i, j] *= -1
            after = model.compute_energy() * N * N
            model.spins[i, j] *= -1
            assert after - before == pytest.approx(dE)


def test_low_temperature_stays_magnetized() -> None:
    model = CPUIsing2D(N=16, T=0.1, init="up")
    model.metropolis_sweep(100)
    assert model.compute_magnetization() > 0.95


def test_high_temperature_is_disordered() -> None:
    model = CPUIsing2D(N=16, T=10.0, init="up")
    model.metropolis_sweep(100)
    assert model.compute_magnetization() < 0.25
