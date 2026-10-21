"""HDF5 persistence for temperature-sweep simulations."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def save_simulation_hdf5(
    filepath: str | Path,
    metadata: dict[str, Any],
    scalar_data: dict[str, np.ndarray],
    configuration_snapshots: np.ndarray,
) -> None:
    """Write scalars (float32), snapshots (gzip int8, one per T) and metadata attrs."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        grp = f.create_group("scalars")
        for key, val in scalar_data.items():
            grp.create_dataset(key, data=np.asarray(val, dtype=np.float32))
        f.create_dataset(
            "snapshots",
            data=np.asarray(configuration_snapshots, dtype=np.int8),
            compression="gzip",
            compression_opts=4,
        )
        for key, val in metadata.items():
            f.attrs[key] = val


def load_simulation_hdf5(filepath: str | Path) -> dict[str, Any]:
    """Return {'metadata': dict, 'scalars': dict of arrays, 'snapshots': int8 array}."""
    with h5py.File(filepath, "r") as f:
        metadata = {
            k: v.item() if isinstance(v, np.generic) else v for k, v in f.attrs.items()
        }
        scalars = {k: d[()] for k, d in f["scalars"].items()}
        snapshots = f["snapshots"][()]
    return {"metadata": metadata, "scalars": scalars, "snapshots": snapshots}
