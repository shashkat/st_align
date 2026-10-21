"""autoalign.py: automatic keypoint placement, checked against the dummy dataset's known transform."""
import sys

import numpy as np
import pytest

import autoalign as aa
import main
from helpers import ORIENT_DUMMY, ArrayImage


@pytest.fixture(scope="module")
def images(dataset):
    return main.PyramidImage(dataset["xenium"]), main.PyramidImage(dataset["if_path"])


def keypoint_errors(result, A):
    """Distance (IF px) of each placed IF point from where the true transform puts its Xenium point."""
    X = np.array([p[0] for p in result["pairs"]])
    U = np.array([p[1] for p in result["pairs"]])
    return np.linalg.norm(U - aa._apply(A, X), axis=1)


@pytest.mark.parametrize("orient, mirrored", [(ORIENT_DUMMY, False), (np.eye(3), True)])
def test_recovers_true_transform(images, true_A, orient, mirrored):
    """Right way round: matches directly. Mirrored display: matches after trying the mirror image."""
    r = aa.auto_keypoints(*images, orient, "gray", n_keypoints=10)
    assert r["mirrored"] is mirrored
    assert len(r["pairs"]) == 10
    assert keypoint_errors(r, true_A).max() < 0.5
    assert np.allclose(r["A"][:2, :2], true_A[:2, :2], atol=0.01)


def test_keypoints_are_spread_out(images):
    r = aa.auto_keypoints(*images, ORIENT_DUMMY, "gray", n_keypoints=6)
    X = np.array([p[0] for p in r["pairs"]])
    d = np.linalg.norm(X[:, None] - X[None], axis=2) + np.eye(len(X)) * 1e9
    assert d.min() > 150                             # Xenium px


def test_noise_fails_cleanly():
    rng = np.random.default_rng(0)
    xen = ArrayImage(rng.integers(0, 1000, (600, 800)).astype(np.uint16))
    ifim = ArrayImage(rng.integers(0, 255, (500, 700, 3)).astype(np.uint8))
    with pytest.raises(aa.AutoAlignError):
        aa.auto_keypoints(xen, ifim, np.eye(3))


def test_missing_opencv_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(aa.AutoAlignError, match="opencv-python-headless"):
        aa._cv2()


def test_orient_matrix_maps_pixels():
    a = np.zeros((30, 50), np.float32)
    a[3, 7] = 1                                       # (x=7, y=3)
    for O in (main.FLIP_H, main.FLIP_V, main.ROT_CW, main.ROT_CW @ main.FLIP_V):
        out, M = aa.orient(a, O)
        x, y = np.round(aa._apply(M, [[7, 3]])[0]).astype(int)
        assert out[y, x] == 1 and out.sum() == 1


def test_to_gray_modes():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (90, 40, 160)                        # haematoxylin-like purple
    rgb[1, 1] = (250, 250, 250)                      # background
    h = aa.to_gray(rgb, "hematoxylin")
    assert h[0, 0] > 0.3 and h[1, 1] < 0.05
    assert aa.to_gray(rgb, "b")[0, 0] == 160
    assert np.isclose(aa.to_gray(rgb, "gray")[0, 0], 290 / 3)
    assert aa.to_gray(np.ones((3, 3), np.uint16), "hematoxylin").dtype == np.float32   # 1 channel: as is


def test_clip_u8_and_farthest_point_order():
    out = aa.clip_u8(np.arange(100, dtype=float).reshape(10, 10))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    pts = np.array([[0, 0], [1, 0], [100, 0], [0, 100], [100, 100]], float)
    order = aa.farthest_point_order(pts)
    assert sorted(order) == list(range(5))
    assert 1 not in order[:4]                        # the near-duplicate point comes last
