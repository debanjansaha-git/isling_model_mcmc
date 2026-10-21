"""Publication figures for the N=128 temperature sweep."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.cpu_ising import T_C
from src.data_io import load_simulation_hdf5

DATA_PATH = "data/ising_sweep_N128.h5"
SNAPSHOT_TEMPS = (1.2, 2.27, 3.8)


def plot_phase_transitions(data: dict, out: str = "docs/images/phase_transitions.png") -> None:
    sc = data["scalars"]
    N = data["metadata"]["N"]
    panels = [
        ("magnetization", r"$\langle |M| \rangle$", "tab:blue"),
        ("energy", r"$\langle E \rangle$", "tab:orange"),
        ("susceptibility", r"$\chi$", "tab:green"),
        ("specific_heat", r"$C_v$", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, (key, label, color) in zip(axes.flat, panels):
        ax.plot(sc["T"], sc[key], "o-", ms=3, lw=1, color=color)
        ax.axvline(T_C, color="red", ls="--", lw=1.2, label=rf"$T_c \approx {T_C:.4f}$")
        if key in ("susceptibility", "specific_heat"):
            tp = sc["T"][np.argmax(sc[key])]
            ax.plot(tp, sc[key].max(), "k*", ms=10, label=f"peak at T={tp:.2f}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Temperature T")
    fig.suptitle(f"2D Ising model, {N}×{N} lattice (Metropolis, GPU checkerboard)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_domain_snapshots(data: dict, out: str = "docs/images/domain_snapshots.png") -> None:
    T = data["scalars"]["T"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    cmap = ListedColormap(["#2c3e50", "#f1c40f"])  # -1, +1
    for ax, target, title in zip(
        axes, SNAPSHOT_TEMPS, ("ordered domains", "critical clusters", "thermal noise")
    ):
        idx = int(np.argmin(np.abs(T - target)))
        ax.imshow(data["snapshots"][idx], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"T = {T[idx]:.2f}  ({title})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_simulation_hdf5(DATA_PATH)
    plot_phase_transitions(data)
    plot_domain_snapshots(data)
    print("wrote docs/images/phase_transitions.png and docs/images/domain_snapshots.png")

    from src.report import write_report

    write_report(data)
    print("wrote SIMULATION_REPORT.md")


if __name__ == "__main__":
    main()
