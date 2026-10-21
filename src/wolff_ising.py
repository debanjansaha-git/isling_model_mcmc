"""Single-cluster Wolff algorithm for the 2D Ising model (CPU, stack-based DFS)."""

import numpy as np

from src.cpu_ising import CPUIsing2D

_POOL_SIZE = 8192


class WolffIsing2D(CPUIsing2D):
    """Wolff cluster updates on an N x N periodic lattice.

    A seed spin is chosen at random and neighbors with equal spin join the cluster
    with probability P_add = 1 - exp(-2 J / T); the whole cluster is then flipped.
    For h != 0 the J-part proposal is unchanged and the flip is accepted with
    min(1, exp(-dE_field / T)), dE_field = 2 h s |C|, which preserves detailed balance.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_cluster: list[int] = []
        self._pool: list[float] = []
        self._pi = 0

    def cluster_update(self, compute_energy: bool = True) -> tuple[int, float]:
        """One cluster move. Returns (proposed cluster size, total energy change).

        The energy change is 0 if the move was rejected (field only) and NaN when
        compute_energy is False. `last_cluster` holds the flipped flat indices
        (empty if rejected).
        """
        N = self.N
        n = N * N
        flat = self.spins.reshape(-1)  # view: the lattice is C-contiguous
        spins = memoryview(flat)  # fast scalar access returning Python ints
        p_add = 1.0 - np.exp(-2.0 * self.J / self.T)

        seed = int(np.random.randint(n))
        s = spins[seed]
        in_cluster = bytearray(n)
        in_cluster[seed] = 1
        members = [seed]
        stack = [seed]
        pool, pi = self._pool, self._pi
        while stack:
            r, c = divmod(stack.pop(), N)
            for nb in (((r + 1) % N) * N + c, ((r - 1) % N) * N + c,
                       r * N + (c + 1) % N, r * N + (c - 1) % N):
                if not in_cluster[nb] and spins[nb] == s:
                    if pi >= len(pool):
                        pool, pi = np.random.random(_POOL_SIZE).tolist(), 0
                    if pool[pi] < p_add:
                        in_cluster[nb] = 1
                        members.append(nb)
                        stack.append(nb)
                    pi += 1
        self._pool, self._pi = pool, pi
        size = len(members)

        dE_field = 2.0 * self.h * s * size
        if dE_field > 0 and np.random.random() >= np.exp(-dE_field / self.T):
            self.last_cluster = []
            return size, 0.0

        dE = float("nan")
        if compute_energy:
            # Boundary bonds: aligned with the cluster -> broken (+2J), else repaired (-2J).
            dE = dE_field
            for i in members:
                r, c = divmod(i, N)
                for nb in (((r + 1) % N) * N + c, ((r - 1) % N) * N + c,
                           r * N + (c + 1) % N, r * N + (c - 1) % N):
                    if not in_cluster[nb]:
                        dE += 2.0 * self.J if spins[nb] == s else -2.0 * self.J
        flat[members] = -s
        self.last_cluster = members
        return size, dE

    def run(
        self, n_updates: int, record_energy: bool = False
    ) -> tuple[np.ndarray, ...]:
        """Perform n_updates cluster moves, recording |M| per spin and cluster size.

        Returns (mags, sizes) or (mags, sizes, energies per spin) if record_energy.
        """
        N2 = self.N**2
        mags = np.empty(n_updates)
        sizes = np.empty(n_updates, dtype=np.int64)
        energies = np.empty(n_updates)
        e_total = self.compute_energy() * N2
        for t in range(n_updates):
            sizes[t], dE = self.cluster_update(compute_energy=record_energy)
            mags[t] = self.compute_magnetization()
            if record_energy:
                e_total += dE
                energies[t] = e_total / N2
        return (mags, sizes, energies) if record_energy else (mags, sizes)


def demo(N: int = 64, updates: int = 2000) -> None:
    for T in (1.5, 2.269, 3.5):
        np.random.seed(0)
        model = WolffIsing2D(N=N, T=T, init="random")
        model.run(500)
        mags, sizes, energies = model.run(updates, record_energy=True)
        print(f"T={T}: <|M|>={mags.mean():.3f} <E>={energies.mean():.3f} "
              f"<cluster>/N^2={sizes.mean() / N**2:.3f}")


if __name__ == "__main__":
    demo()
