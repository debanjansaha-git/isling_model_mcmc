"""Finite-size-scaling figures: Binder cumulant crossings and susceptibility collapse."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cpu_ising import T_C
from src.data_io import load_fss_hdf5

DATA_PATH = "data/fss_analysis.h5"
GAMMA, NU = 1.75, 1.0  # exact 2D Ising exponents
COLORS = {32: "tab:blue", 64: "tab:orange", 128: "tab:green"}


def plot_binder_cumulants(data: dict, out: str = "docs/images/binder_cumulants.png") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, r in sorted(data["results"].items()):
        ax.errorbar(r["T"], r["binder"], yerr=r["binder_err"], fmt="o-", ms=3, lw=1,
                    capsize=2, color=COLORS.get(N), label=f"N = {N}")
    ax.axvline(T_C, color="red", ls="--", lw=1.2, label=rf"$T_c \approx {T_C:.4f}$")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"Binder cumulant $U_4$")
    ax.set_title("Binder cumulant crossings (jackknife errors)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_data_collapse(data: dict, out: str = "docs/images/fss_data_collapse.png") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, r in sorted(data["results"].items()):
        x = (r["T"] - T_C) * N ** (1 / NU)
        scale = N ** (-GAMMA / NU)
        ax.errorbar(x, r["susceptibility"] * scale, yerr=r["susceptibility_err"] * scale,
                    fmt="o", ms=3, capsize=2, color=COLORS.get(N), label=f"N = {N}")
    ax.set_xlabel(r"$(T - T_c)\,N^{1/\nu}$")
    ax.set_ylabel(r"$\chi\,N^{-\gamma/\nu}$")
    ax.set_title(rf"Susceptibility data collapse ($\gamma$={GAMMA}, $\nu$={NU})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_fss_hdf5(DATA_PATH)
    plot_binder_cumulants(data)
    plot_data_collapse(data)
    print("wrote docs/images/binder_cumulants.png and docs/images/fss_data_collapse.png")


if __name__ == "__main__":
    main()
