"""Boundaries: nucleus polygons from parquet (µm -> Xenium px) and fast spatial selection."""
import numpy as np
import pandas as pd
import pytest

import dummy_data
import main


@pytest.fixture(scope="module")
def bounds(dataset):
    return main.Boundaries(dataset["boundaries"], dummy_data.PX_UM)


def test_cells_and_centroids(bounds, dataset):
    assert len(bounds.cx) == dataset["n_nuclei"]
    centroids = np.column_stack([bounds.cx, bounds.cy])
    # every generated centre has a centroid within a pixel
    d = np.linalg.norm(centroids[:, None, :] - np.asarray(dataset["centres"])[None, :200, :], axis=2).min(0)
    assert d.max() < 1.0


def test_typical_size_is_nucleus_diameter(bounds):
    assert 15 < bounds.typical_size < 35         # ellipse semi-axes 7-17 px


def test_select_matches_brute_force(bounds, dataset):
    df = pd.read_parquet(dataset["boundaries"])
    x, y = df.vertex_x.to_numpy(float) / dummy_data.PX_UM, df.vertex_y.to_numpy(float) / dummy_data.PX_UM
    box = (900, 700, 1700, 1300)
    (pts, connect), n = bounds.select(*box)
    ids = df.cell_id.to_numpy()
    cx = pd.Series(x).groupby(ids).mean()
    cy = pd.Series(y).groupby(ids).mean()
    inside = cx[(cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])].index
    assert n == len(inside) > 0
    assert len(pts) == n * dummy_data.VERTS
    # one polyline per cell: connect is False exactly on each cell's last vertex
    assert (~connect).sum() == n and not connect[dummy_data.VERTS - 1] and connect[: dummy_data.VERTS - 1].all()
    sel = np.isin(ids, inside)
    assert np.allclose(np.sort(pts[:, 0]), np.sort(x[sel]))


def test_select_empty_and_over_cap(bounds, monkeypatch):
    assert bounds.select(-500, -500, -100, -100) == (None, 0)
    monkeypatch.setattr(main, "MAX_CELLS", 10)
    res, n = bounds.select(0, 0, 3000, 2400)
    assert res is None and n > 10
