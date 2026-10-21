from pathlib import Path

import numpy as np
import pytest
import torch

from src.cpu_ising import T_C
from src.data_io import load_simulation_hdf5
from src.simulate import run_simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PRODUCTION_FILE = Path("data/ising_sweep_N128.h5")


def test_susceptibility_peaks_near_tc_reduced_run() -> None:
    temps = np.round(np.arange(1.5, 3.51, 0.05), 2)
    res = run_simulation(N=32, temps=temps, therm_sweeps=500, meas_sweeps=3000, seed=1)
    chi, M = res["scalars"]["susceptibility"], res["scalars"]["magnetization"]
    T = res["scalars"]["T"]

    assert abs(T[np.argmax(chi)] - T_C) <= 0.10
    assert M[0] > 0.98  # T = 1.5: ordered
    assert M[-1] < 0.2  # T = 3.5: disordered
    assert res["snapshots"].shape == (len(temps), 32, 32)


@pytest.mark.skipif(not PRODUCTION_FILE.exists(), reason="run src.simulate first")
def test_production_susceptibility_peaks_near_tc() -> None:
    out = load_simulation_hdf5(PRODUCTION_FILE)
    T, chi = out["scalars"]["T"], out["scalars"]["susceptibility"]
    assert abs(T[np.argmax(chi)] - T_C) <= 0.10
