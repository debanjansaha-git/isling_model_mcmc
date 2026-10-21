"""GPU replica-exchange (parallel tempering) MCMC built on the checkerboard engine."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.gpu_ising import GPUIsing2D

OUTPUT_PATH = "data/pt_demo.json"


def swap_probability(E_i: float, E_j: float, T_i: float, T_j: float) -> float:
    """Metropolis acceptance for exchanging configurations between temperatures.

    P = min(1, exp((1/T_i - 1/T_j)(E_i - E_j))), E_* total energies of the
    configurations currently at T_i and T_j.
    """
    return float(min(1.0, np.exp(min((1.0 / T_i - 1.0 / T_j) * (E_i - E_j), 0.0))))


class ParallelTempering:
    """R replicas of an N x N lattice, slot k always running at temps[k] (ascending).

    Local updates use the batched checkerboard sweep. Swaps exchange whole
    configurations between adjacent slots (alternating even/odd pairs), so each
    slot keeps its own fixed temperature.
    """

    def __init__(
        self,
        N: int,
        temps: np.ndarray,
        init: str = "random",
        J: float = 1.0,
        device: str = "cuda:0",
        ensembles: int = 1,
    ) -> None:
        self.N = N
        self.temps = np.asarray(temps, dtype=np.float64)
        self.R = len(self.temps)
        self.E = ensembles  # independent ladders run side by side (ensemble-major layout)
        self.model = GPUIsing2D(N=N, T=np.tile(self.temps, ensembles), batch=self.R * ensembles,
                                J=J, init=init, device=device)
        self._swap_count = 0  # drives even/odd pair alternation across calls
        self._sweeps_done = 0
        dev = self.model.device
        self.beta = torch.as_tensor(1.0 / self.temps, dtype=torch.float64, device=dev)
        self.swap_attempts = torch.zeros(self.R - 1, dtype=torch.int64, device=dev)
        self.swap_accepts = torch.zeros(self.R - 1, dtype=torch.int64, device=dev)

    @torch.no_grad()
    def swap_step(self, parity: int) -> None:
        """Propose swaps for slot pairs (i, i+1) with i % 2 == parity, all at once."""
        dev = self.model.device
        i = torch.arange(parity, self.R - 1, 2, device=dev)
        if len(i) == 0:
            return
        E = (self.model.compute_energy() * self.N**2).reshape(self.E, self.R)  # float64
        delta = (self.beta[i] - self.beta[i + 1]) * (E[:, i] - E[:, i + 1])
        prob = torch.exp(torch.clamp(delta, max=0.0))
        accept = torch.rand(prob.shape, dtype=torch.float64, device=dev) < prob  # (E, pairs)

        perm = torch.arange(self.R, device=dev).repeat(self.E, 1)  # (E, R), within-ladder
        e_idx, k_idx = accept.nonzero(as_tuple=True)
        a = i[k_idx]
        perm[e_idx, a], perm[e_idx, a + 1] = a + 1, a  # exchange accepted pairs at once
        offset = torch.arange(self.E, device=dev)[:, None] * self.R
        self.model.spins = self.model.spins[(perm + offset).reshape(-1)]
        self.swap_attempts[i] += self.E
        self.swap_accepts[i] += accept.sum(0)

    def sweep(self, n_sweeps: int = 1, swap_interval: int = 1) -> None:
        """n_sweeps local sweeps, with a swap proposal after every swap_interval."""
        for _ in range(n_sweeps):
            self.model.sweep(1)
            self._sweeps_done += 1
            if self._sweeps_done % swap_interval == 0:
                self.swap_step(parity=self._swap_count % 2)
                self._swap_count += 1

    def run(self, therm_sweeps: int, meas_sweeps: int, swap_interval: int = 1) -> dict:
        """Thermalize, then average |M| and E per slot over meas_sweeps sweeps."""
        self.sweep(therm_sweeps, swap_interval)
        self.swap_attempts.zero_()
        self.swap_accepts.zero_()
        dev = self.model.device
        m_sum = torch.zeros(self.E * self.R, dtype=torch.float64, device=dev)
        e_sum = torch.zeros_like(m_sum)
        for _ in range(meas_sweeps):
            self.sweep(1, swap_interval)
            m_sum += self.model.compute_magnetization()
            e_sum += self.model.compute_energy()
        rate = self.swap_accepts.double() / self.swap_attempts.clamp(min=1)
        by_ensemble = (m_sum / meas_sweeps).reshape(self.E, self.R)
        return {
            "T": self.temps,
            "magnetization": by_ensemble.mean(0).cpu().numpy(),
            "energy": (e_sum / meas_sweeps).reshape(self.E, self.R).mean(0).cpu().numpy(),
            "swap_rate": rate.cpu().numpy(),
            "final_magnetization": self.model.compute_magnetization()
            .reshape(self.E, self.R).cpu().numpy(),
            "window_magnetization": by_ensemble.cpu().numpy(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel tempering vs plain Metropolis")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=151)
    parser.add_argument("--therm", type=int, default=10000)
    parser.add_argument("--meas", type=int, default=5000)
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    temps = np.geomspace(1.0, 4.0, args.replicas)

    t0 = time.perf_counter()
    pt = ParallelTempering(N=args.N, temps=temps, init="random")
    res = pt.run(args.therm, args.meas)
    pt_time = time.perf_counter() - t0

    # Control: the same number of independent random-start chains at T = 1, no swaps.
    plain = GPUIsing2D(N=args.N, T=1.0, batch=args.replicas, init="random")
    plain.sweep(args.therm + args.meas)
    m_plain = plain.compute_magnetization().cpu().numpy()
    stuck = float((m_plain < 0.9).mean())

    out = {
        "N": args.N, "replicas": args.replicas, "T_min": 1.0, "T_max": 4.0,
        "therm_sweeps": args.therm, "meas_sweeps": args.meas,
        "pt_magnetization_T1": float(res["magnetization"][0]),
        "pt_energy_T1": float(res["energy"][0]),
        "mean_swap_rate": float(res["swap_rate"].mean()),
        "min_swap_rate": float(res["swap_rate"].min()),
        "plain_stuck_fraction": stuck,
        "plain_mean_magnetization": float(m_plain.mean()),
        "runtime_s": pt_time,
        "temps": [float(x) for x in temps],
        "swap_rate": [float(x) for x in res["swap_rate"]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"PT, N={args.N}, {args.replicas} replicas T in [1, 4], {pt_time:.0f} s")
    print(f"  T=1 slot: <|M|>={out['pt_magnetization_T1']:.4f}, <E>={out['pt_energy_T1']:.4f}")
    print(f"  swap acceptance: mean {out['mean_swap_rate']:.2f}, min {out['min_swap_rate']:.2f}")
    print(f"Plain Metropolis from random start at T=1 ({args.replicas} chains, "
          f"{args.therm + args.meas} sweeps): {stuck:.0%} stuck below |M|=0.9, "
          f"mean |M|={out['plain_mean_magnetization']:.3f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
