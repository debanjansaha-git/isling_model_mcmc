import h5py
import numpy as np

from src.data_io import load_simulation_hdf5, save_simulation_hdf5


def _payload() -> tuple[dict, dict, np.ndarray]:
    T = np.linspace(1.0, 4.0, 7)
    scalars = {
        "T": T,
        "magnetization": np.linspace(1, 0, 7),
        "energy": np.linspace(-2, -0.5, 7),
        "susceptibility": np.arange(7.0),
        "specific_heat": np.arange(7.0) / 2,
    }
    metadata = {"N": 8, "therm_sweeps": 10, "device": "cuda:0", "J": 1.0}
    snaps = np.random.default_rng(0).choice([-1, 1], size=(7, 8, 8)).astype(np.int8)
    return metadata, scalars, snaps


def test_roundtrip_preserves_data(tmp_path) -> None:
    metadata, scalars, snaps = _payload()
    path = tmp_path / "sub" / "sim.h5"  # parent directory must be created
    save_simulation_hdf5(path, metadata, scalars, snaps)

    out = load_simulation_hdf5(path)
    assert out["metadata"]["N"] == 8
    assert out["metadata"]["device"] == "cuda:0"
    assert out["metadata"]["J"] == 1.0
    for key, val in scalars.items():
        np.testing.assert_allclose(out["scalars"][key], val, rtol=1e-6)
    assert np.array_equal(out["snapshots"], snaps)


def test_storage_dtypes_and_compression(tmp_path) -> None:
    metadata, scalars, snaps = _payload()
    path = tmp_path / "sim.h5"
    save_simulation_hdf5(path, metadata, scalars, snaps)

    with h5py.File(path, "r") as f:
        for key in scalars:
            assert f[f"scalars/{key}"].dtype == np.float32
        assert f["snapshots"].dtype == np.int8
        assert f["snapshots"].compression == "gzip"

    out = load_simulation_hdf5(path)
    assert out["snapshots"].dtype == np.int8
