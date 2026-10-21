"""CPU baseline: single-spin Metropolis-Hastings for the 2D Ising model."""

import numpy as np

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))


class CPUIsing2D:
    """N x N Ising lattice with periodic boundaries (J, k_B = 1 units)."""

    def __init__(
        self,
        N: int = 32,
        T: float = 2.0,
        J: float = 1.0,
        h: float = 0.0,
        init: str = "random",
    ) -> None:
        if init not in ("random", "up"):
            raise ValueError("init must be 'random' or 'up'")
        self.N = N
        self.T = T
        self.J = J
        self.h = h
        if init == "random":
            self.spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(N, N))
        else:
            self.spins = np.ones((N, N), dtype=np.int8)

    def compute_energy(self) -> float:
        """Energy per spin; each bond counted once via right/down rolls."""
        s = self.spins.astype(np.int64)
        bonds = np.sum(s * np.roll(s, -1, axis=0)) + np.sum(s * np.roll(s, -1, axis=1))
        return float((-self.J * bonds - self.h * s.sum()) / self.N**2)

    def compute_magnetization(self) -> float:
        """Absolute magnetization per spin |M|."""
        return float(abs(self.spins.astype(np.int64).sum()) / self.N**2)

    def delta_energy(self, i: int, j: int) -> float:
        """Total-energy change if spin (i, j) were flipped."""
        N, s = self.N, self.spins
        nb = (
            int(s[(i + 1) % N, j])
            + int(s[(i - 1) % N, j])
            + int(s[i, (j + 1) % N])
            + int(s[i, (j - 1) % N])
        )
        return 2.0 * int(s[i, j]) * (self.J * nb + self.h)

    def metropolis_sweep(self, num_sweeps: int = 1) -> None:
        """Run num_sweeps sweeps; each is N^2 random single-spin proposals."""
        N = self.N
        for _ in range(num_sweeps):
            sites = np.random.randint(0, N, size=(N * N, 2))
            rand = np.random.random(N * N)
            for (i, j), r in zip(sites, rand):
                dE = self.delta_energy(i, j)
                if dE <= 0 or r < np.exp(-dE / self.T):
                    self.spins[i, j] *= -1


def demo(N: int = 32, sweeps: int = 500, temps: tuple[float, ...] = (1.5, 3.5)) -> None:
    print(f"T_c = {T_C:.6f}")
    for T in temps:
        np.random.seed(42)
        model = CPUIsing2D(N=N, T=T, init="up")  # cold start avoids frozen stripe domains
        mags, ens = [], []
        for _ in range(sweeps):
            model.metropolis_sweep(1)
            mags.append(model.compute_magnetization())
            ens.append(model.compute_energy())
        half = sweeps // 2  # discard first half as equilibration
        print(
            f"T={T}: <|M|>={np.mean(mags[half:]):.4f}  <E>={np.mean(ens[half:]):.4f}"
            f"  (N={N}, {sweeps} sweeps, last {sweeps - half} averaged)"
        )


if __name__ == "__main__":
    demo()
