import itertools

import numpy as np
import pytest
import torch

from src.gpu_ising import GPUIsing2D
from src.parallel_tempering import ParallelTempering, swap_probability
from src.wolff_ising import WolffIsing2D

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)
    torch.manual_seed(0)


# ---- Wolff cluster algorithm ------------------------------------------------

@pytest.mark.parametrize("h", [0.0, 0.7])
def test_wolff_incremental_energy_matches_recomputed(h: float) -> None:
    model = WolffIsing2D(N=8, T=2.3, h=h, init="random")
    for _ in range(300):
        e_before = model.compute_energy() * model.N**2
        _, dE = model.cluster_update()
        e_after = model.compute_energy() * model.N**2
        assert e_after - e_before == pytest.approx(dE, abs=1e-9)


def test_wolff_zero_temperature_flips_whole_lattice_and_conserves_energy() -> None:
    model = WolffIsing2D(N=6, T=0.05, init="up")  # P_add = 1 - exp(-40) ~ 1
    size, dE = model.cluster_update()
    assert size == 36
    assert dE == 0.0
    assert model.compute_energy() == pytest.approx(-2.0)
    assert np.all(model.spins == -1)


def test_wolff_cluster_is_connected_and_single_valued() -> None:
    model = WolffIsing2D(N=10, T=2.3, init="random")
    N = model.N
    for _ in range(50):
        before = model.spins.copy()
        size, _ = model.cluster_update()
        members = set(model.last_cluster)
        assert len(members) == size
        assert len({int(before.flat[i]) for i in members}) == 1  # one sign
        assert np.all(model.spins.flat[list(members)] == -before.flat[list(members)])
        outside = np.ones(N * N, bool)
        outside[list(members)] = False
        assert np.array_equal(model.spins.reshape(-1)[outside], before.reshape(-1)[outside])
        # connectivity on the torus
        seen, stack = {next(iter(members))}, [next(iter(members))]
        while stack:
            r, c = divmod(stack.pop(), N)
            for nb in (((r + 1) % N) * N + c, ((r - 1) % N) * N + c,
                       r * N + (c + 1) % N, r * N + (c - 1) % N):
                if nb in members and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == members


@pytest.mark.parametrize("h", [0.0, 0.5])
def test_wolff_samples_exact_boltzmann_distribution_4x4(h: float) -> None:
    N, T = 4, 2.5
    states = np.array(list(itertools.product([-1, 1], repeat=N * N)), dtype=np.int8)
    states = states.reshape(-1, N, N)
    s = states.astype(np.int64)
    bonds = (s * np.roll(s, -1, 1)).sum((1, 2)) + (s * np.roll(s, -1, 2)).sum((1, 2))
    mag = s.sum((1, 2))
    H = -bonds - h * mag
    w = np.exp(-(H - H.min()) / T)
    w /= w.sum()
    exact_E = (w * H).sum() / N**2
    exact_M = (w * np.abs(mag)).sum() / N**2

    model = WolffIsing2D(N=N, T=T, h=h, init="random")
    model.run(500)
    mags, _, energies = model.run(40_000, record_energy=True)
    assert energies.mean() == pytest.approx(exact_E, abs=0.02)
    assert mags.mean() == pytest.approx(exact_M, abs=0.02)


# ---- GPU parallel tempering ---------------------------------------------------

def test_swap_probability_satisfies_detailed_balance() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        Ti, Tj = np.sort(rng.uniform(1.0, 4.0, 2))
        Ei, Ej = rng.uniform(-1000, -900, 2)  # keeps exp(...) away from underflow
        a_fwd = swap_probability(Ei, Ej, Ti, Tj)
        a_rev = swap_probability(Ej, Ei, Ti, Tj)  # state after the swap
        assert 0.0 <= a_fwd <= 1.0
        # pi(x_i@Ti, x_j@Tj) A(fwd) == pi(x_j@Ti, x_i@Tj) A(rev), in log space
        lhs = np.log(a_fwd) - Ei / Ti - Ej / Tj
        rhs = np.log(a_rev) - Ej / Ti - Ei / Tj
        assert lhs == pytest.approx(rhs, abs=1e-9)
    # cold replica with higher energy than the hot one: swap always accepted
    assert swap_probability(-100.0, -300.0, 1.0, 2.0) == 1.0


@needs_cuda
def test_swap_step_permutes_configurations() -> None:
    pt = ParallelTempering(N=8, temps=np.linspace(2.0, 3.0, 6))

    def multiset() -> list[bytes]:
        return sorted(r.cpu().numpy().tobytes() for r in pt.model.spins)

    before = multiset()
    for parity in (0, 1) * 5:
        pt.swap_step(parity)
        assert multiset() == before
    assert pt.swap_attempts.sum() > 0


@needs_cuda
def test_parallel_tempering_matches_independent_chains() -> None:
    temps = np.array([2.0, 2.5, 3.0, 3.5])
    pt = ParallelTempering(N=8, temps=temps)
    res = pt.run(therm_sweeps=1000, meas_sweeps=10_000)

    plain = GPUIsing2D(N=8, T=temps, batch=4, init="random")
    plain.sweep(1000)
    e = torch.zeros(4, dtype=torch.float64, device=plain.device)
    for _ in range(10_000):
        plain.sweep(1)
        e += plain.compute_energy()
    np.testing.assert_allclose(res["energy"], (e / 10_000).cpu().numpy(), atol=0.04)
    assert 0.05 < res["swap_rate"].mean() <= 1.0


@needs_cuda
def test_parallel_tempering_escapes_frozen_stripes() -> None:
    temps = np.geomspace(1.0, 4.0, 32)
    pt = ParallelTempering(N=16, temps=temps, init="random")
    res = pt.run(therm_sweeps=6000, meas_sweeps=3000)
    assert res["T"][0] == pytest.approx(1.0)
    assert res["magnetization"][0] > 0.98
