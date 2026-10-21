"""PyramidImage: lazy reading of pyramidal OME-TIFFs (multi-channel uint16 and RGB)."""
import numpy as np

import main


def test_xenium_image_metadata(dataset):
    img = main.PyramidImage(dataset["xenium"])
    assert img.axes == "CYX"
    assert img.n_channels == 2 and img.channel_names == ["DAPI", "Membrane"]
    assert img.pixel_size == 0.2125
    assert img.shapes == [(2400, 3000), (1200, 1500), (600, 750)]
    assert np.allclose(img.ds, [(1, 1), (2, 2), (4, 4)])
    assert not img.rgb


def test_if_image_is_rgb_pyramid(dataset):
    img = main.PyramidImage(dataset["if_path"])
    assert img.rgb and img.axes == "YXS"
    assert len(img.shapes) == 3
    tile = img.read(0, 100, 164, 200, 232)
    assert tile.shape == (64, 32, 3) and tile.dtype == np.uint8


def test_read_follows_selected_channel(dataset):
    img = main.PyramidImage(dataset["xenium"])
    a = img.read(1, 0, 1200, 0, 1500)
    img.channel = 1
    b = img.read(1, 0, 1200, 0, 1500)
    assert a.shape == b.shape == (1200, 1500)
    assert not np.array_equal(a, b)


def test_levels_show_the_same_content(dataset):
    """Level 1 is a 2x downsample of level 0 (same place, half the pixels)."""
    img = main.PyramidImage(dataset["xenium"])
    full = img.read(0, 800, 1200, 1000, 1400).astype(float)
    half = img.read(1, 400, 600, 500, 700).astype(float)
    assert np.corrcoef(full[::2, ::2].ravel(), half.ravel())[0, 1] > 0.9


def test_overview_is_small_and_scaled(dataset):
    img = main.PyramidImage(dataset["xenium"])
    data, (dx, dy) = img.overview()
    assert max(data.shape) <= main.MAX_OVERVIEW
    assert np.isclose(data.shape[1] * dx, 3000) and np.isclose(data.shape[0] * dy, 2400)
