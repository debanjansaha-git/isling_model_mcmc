"""GPU engine: batched checkerboard Metropolis for the 2D Ising model."""

import argparse
import time

import numpy as np
import torch

from src.cpu_ising import T_C

# Float32 temporaries (rand, dE, exp) are built per chunk of at most this many
# spins (~128 MB each), keeping total VRAM well under the 1 GB budget.
DEFAULT_MAX_CHUNK_SPINS = 1 << 25


class GPUIsing2D:
    """B independent N x N Ising lattices, each at its own temperature.

    Spins are stored as int8 of shape (B, N, N). Periodic boundaries via
    torch.roll; Red ((i+j)%2==0) then Black sites are updated in parallel.
    """

    def __init__(
        self,
        N: int = 32,
        T: float | torch.Tensor | np.ndarray = 2.0,
        batch: int = 1,
        J: float = 1.0,
        h: float = 0.0,
        init: str = "random",
        spins: torch.Tensor | np.ndarray | None = None,
        device: str = "cuda:0",
        max_chunk_spins: int = DEFAULT_MAX_CHUNK_SPINS,
    ) -> None:
        if init not in ("random", "up"):
            raise ValueError("init must be 'random' or 'up'")
        self.N, self.B, self.J, self.h = N, batch, J, h
        self.device = torch.device(device)
        self.max_chunk_spins = max_chunk_spins

        T = torch.as_tensor(T, dtype=torch.float32, device=self.device)
        self.T = T.expand(batch).clone().reshape(batch, 1, 1)  # (B, 1, 1)

        if spins is not None:
            self.spins = torch.as_tensor(spins).to(self.device, torch.int8).clone()
            if self.spins.shape != (batch, N, N):
                raise ValueError(f"spins must have shape {(batch, N, N)}")
        elif init == "up":
            self.spins = torch.ones(batch, N, N, dtype=torch.int8, device=self.device)
        else:
            r = torch.randint(0, 2, (batch, N, N), dtype=torch.int8, device=self.device)
            self.spins = 2 * r - 1

        idx = torch.arange(N, device=self.device)
        parity = (idx[:, None] + idx[None, :]) % 2
        self.red = parity == 0
        self.black = parity == 1

    def compute_energy(self) -> torch.Tensor:
        """Energy per spin, shape (B,), float64."""
        s = self.spins
        bonds = (s * torch.roll(s, -1, 1)).sum((1, 2), dtype=torch.int64) + (
            s * torch.roll(s, -1, 2)
        ).sum((1, 2), dtype=torch.int64)
        total = s.sum((1, 2), dtype=torch.int64)
        return (-self.J * bonds.double() - self.h * total.double()) / self.N**2

    def compute_magnetization(self) -> torch.Tensor:
        """Absolute magnetization per spin |M|, shape (B,), float64."""
        total = self.spins.sum((1, 2), dtype=torch.int64)
        return total.abs().double() / self.N**2

    def _half_step(self, sp: torch.Tensor, T: torch.Tensor, mask: torch.Tensor) -> None:
        nb = (
            torch.roll(sp, 1, 1) + torch.roll(sp, -1, 1)
            + torch.roll(sp, 1, 2) + torch.roll(sp, -1, 2)
        )
        dE = 2.0 * sp.float() * (self.J * nb.float() + self.h)
        accept = torch.rand_like(dE) < torch.exp(-dE / T)  # dE <= 0 always accepts
        flip = accept & mask
        sp.mul_(1 - 2 * flip.to(torch.int8))

    @torch.no_grad()
    def sweep(self, num_sweeps: int = 1) -> None:
        """Run num_sweeps full sweeps (Red half-step, then Black half-step)."""
        chunk = max(1, self.max_chunk_spins // (self.N * self.N))
        for _ in range(num_sweeps):
            for b0 in range(0, self.B, chunk):
                sp = self.spins[b0 : b0 + chunk]  # view: updates are in place
                T = self.T[b0 : b0 + chunk]
                self._half_step(sp, T, self.red)
                self._half_step(sp, T, self.black)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Ising benchmark")
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--lattice", type=int, default=128)
    parser.add_argument("--sweeps", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    T = torch.linspace(1.0, 4.0, args.batch)
    model = GPUIsing2D(N=args.lattice, T=T, batch=args.batch, init="up")
    model.sweep(2)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.sweep(args.sweeps)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flips = args.batch * args.lattice**2 * args.sweeps
    print(f"B={args.batch} N={args.lattice} sweeps={args.sweeps} (T_c={T_C:.4f})")
    print(f"{args.sweeps / dt:.1f} sweeps/s, {flips / dt:.3e} spin-updates/s")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
