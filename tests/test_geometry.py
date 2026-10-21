"""Geometry helpers in main.py: 3x3 matrices act on column vectors [x, y, 1] (x = column, y = row)."""
import math

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

import main


def test_apply_matches_matrix_product():
    M = main.T(5, -3) @ main.Rot(0.4) @ main.S(2, 0.5)
    pts = np.array([[0.0, 0.0], [10.0, 20.0], [-3.5, 7.25]])
    expected = (M @ np.column_stack([pts, np.ones(3)]).T).T[:, :2]
    assert np.allclose(main.apply(M, pts), expected)


def test_qtransform_agrees_with_apply():
    """Qt uses row vectors; qtransform must convert so both map points identically."""
    M = main.T(12, 7) @ main.Rot(-0.3) @ main.S(1.5, 0.75) @ main.FLIP_H
    qt = main.qtransform(M)
    for p in ([0, 0], [100, 50], [-20, 300]):
        q = qt.map(QtCore.QPointF(*p))
        assert np.allclose([q.x(), q.y()], main.apply(M, [p])[0])


def test_rot_is_clockwise_on_screen():
    """With y pointing down, a positive angle turns +x towards +y (clockwise on screen)."""
    assert np.allclose(main.apply(main.Rot(math.pi / 2), [[1, 0]])[0], [0, 1])


@pytest.mark.parametrize("O", [main.FLIP_H, main.FLIP_V, main.ROT_CW])
def test_orientation_matrices_are_orthonormal_without_translation(O):
    assert np.allclose(O[:2, :2] @ O[:2, :2].T, np.eye(2))
    assert np.allclose(O[:2, 2], 0)


def test_bbox_and_rect_corners():
    r = QtCore.QRectF(10, 20, 30, 40)
    assert main.bbox(main.rect_corners(r)) == (10, 20, 40, 60)
