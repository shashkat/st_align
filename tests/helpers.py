"""Helpers for driving the GUI in tests with real (synthetic) Qt mouse / key events."""
import numpy as np
from pyqtgraph.Qt import QtCore, QtTest

import main

Qt = QtCore.Qt

# the display orientation that shows the dummy IF the right way round (TRUE_A is mirrored + ~93°)
ORIENT_DUMMY = main.ROT_CW @ main.FLIP_V


def settle(qtbot, ms=150):
    """Let debounced timers (tile loading, outline selection) fire."""
    qtbot.wait(ms)


def set_view(pane, cx, cy, half):
    """Centre a pane on raw point (cx, cy), showing about 2*half raw px."""
    sc = pane.to_scene([[cx, cy]])[0]
    pane.vb.setRange(QtCore.QRectF(sc[0] - half, sc[1] - half, 2 * half, 2 * half), padding=0)


def widget_pos(pane, raw):
    """Viewport pixel position of a raw image point."""
    sc = pane.to_scene([raw])[0]
    return pane.glw.mapFromScene(pane.vb.mapViewToScene(QtCore.QPointF(*sc)))


def _move_to(viewport, pos):
    # step off and back: moving to where the (test-global) cursor already is sends no move event;
    # pause in between because pyqtgraph rate-limits mouse-move signals
    QtTest.QTest.mouseMove(viewport, pos + QtCore.QPoint(3, 3))
    QtTest.QTest.qWait(40)
    QtTest.QTest.mouseMove(viewport, pos)


def move(qtbot, pane, raw):
    _move_to(pane.glw.viewport(), widget_pos(pane, raw))
    qtbot.wait(20)


def click(qtbot, pane, raw):
    pos = widget_pos(pane, raw)
    _move_to(pane.glw.viewport(), pos)
    QtTest.QTest.mouseClick(pane.glw.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    qtbot.wait(20)


def key(qtbot, widget, k, mod=Qt.KeyboardModifier.NoModifier):
    QtTest.QTest.keyClick(widget, k, mod)
    qtbot.wait(20)


def double_click_list_row(qtbot, lw, row):
    """A real double-click: Qt sends press/release, then press/double-click/release."""
    pos = lw.visualItemRect(lw.item(row)).center()
    QtTest.QTest.mouseClick(lw.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    QtTest.QTest.mouseDClick(lw.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    qtbot.wait(50)


def view_center(pane):
    """Centre of a pane's view, as a raw image point."""
    c = pane.vb.viewRect().center()
    return pane.to_raw([[c.x(), c.y()]])[0]


def tissue_points(dataset, n=3, seed=0):
    """n well-separated nucleus centres (Xenium px), away from the image edges."""
    c = np.asarray(dataset["centres"])
    rng = np.random.default_rng(seed)
    picks = [c[rng.integers(len(c))]]
    for _ in range(n - 1):
        d = np.min([np.linalg.norm(c - p, axis=1) for p in picks], axis=0)
        picks.append(c[int(np.argmax(d))])
    return np.array(picks)


class ArrayImage:
    """Minimal in-memory stand-in for main.PyramidImage (single level), for autoalign unit tests."""

    def __init__(self, data):
        self.data = data
        self.rgb = data.ndim == 3
        self.shapes = [data.shape[:2]]
        self.ds = [(1.0, 1.0)]
        self.channel = 0

    def read(self, level, r0, r1, c0, c1, step=1):
        return self.data[r0:r1:step, c0:c1:step]
