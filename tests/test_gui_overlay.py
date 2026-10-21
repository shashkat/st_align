"""Overlay placement: Xenium nuclei float over the IF, lock on click, keypoint on the next click."""
import math

import numpy as np
import pytest

import main
from helpers import ORIENT_DUMMY, Qt, click, key, move, set_view, settle, tissue_points, view_center

pytestmark = pytest.mark.gui


@pytest.fixture
def placing(window, qtbot):
    """Aligner with the IF shown the right way round and overlay placement switched on."""
    window.R.set_orientation(ORIENT_DUMMY)
    window._redraw_keypoints()
    key(qtbot, window, Qt.Key.Key_O)
    assert window.a_ov.isChecked() and window.g_state == "float"
    return window


def go_to(w, qtbot, x, A):
    """Xenium pane centred on x, IF pane on its true match, and the mouse over that match."""
    u = main.apply(A, [x])[0]
    set_view(w.L, *x, half=300)
    set_view(w.R, *u, half=150)
    settle(qtbot)
    move(qtbot, w.R, u)
    return u


def test_ghost_anchors_xenium_centre_under_the_mouse(placing, dataset, true_A, qtbot):
    x = tissue_points(dataset, 1)[0]
    u = go_to(placing, qtbot, x, true_A)
    assert placing.R.ghost.isVisible() and placing._ghost_ok
    A_prov = placing._current_map()
    upp = float(np.mean(placing.R.vb.viewPixelSize()))
    assert np.linalg.norm(main.apply(A_prov, [view_center(placing.L)])[0] - u) < 2 * upp


def test_ghost_scale_follows_zoom_ratio(placing, dataset, true_A, qtbot):
    x = tissue_points(dataset, 1)[0]
    u = go_to(placing, qtbot, x, true_A)
    s1 = math.sqrt(abs(np.linalg.det(placing._current_map()[:2, :2])))
    set_view(placing.R, *u, half=300)                                    # zoom the IF out 2x
    settle(qtbot)
    move(qtbot, placing.R, u)
    s2 = math.sqrt(abs(np.linalg.det(placing._current_map()[:2, :2])))
    assert np.isclose(s2 / s1, 2, rtol=0.02)


def test_three_overlay_keypoints_recover_true_transform(placing, dataset, true_A, qtbot):
    for i, x in enumerate(tissue_points(dataset, 3)):
        u = go_to(placing, qtbot, x, true_A)
        click(qtbot, placing.R, u)
        assert placing.g_state == "locked"
        click(qtbot, placing.R, u)                                       # keypoint where we're confident
        assert len(placing.pairs) == i + 1 and placing.g_state == "float"
    assert np.allclose(placing.A[:2, :2], true_A[:2, :2], atol=0.01)
    centre = main.apply(true_A, [[1500, 1200]])[0]
    assert np.linalg.norm(main.apply(placing.A, [[1500, 1200]])[0] - centre) < 2
    assert placing.a_link.isChecked() and placing.a_nuc.isChecked()


def test_overlay_keeps_floating_after_fit_and_is_not_synced(placing, dataset, true_A, qtbot):
    """Regression: after 3 keypoints, clicks must still float/lock/place, not drop a pair on the fit."""
    for x in tissue_points(dataset, 3):
        placing._add_pair(x, main.apply(true_A, [x])[0])
    assert placing.g_state == "float"
    x = tissue_points(dataset, 4, seed=3)[-1]
    u = go_to(placing, qtbot, x, true_A)
    off = u + [15.0, -10.0]                                              # lock the ghost off the fit
    move(qtbot, placing.R, off)
    assert placing.R.ghost.isVisible()
    click(qtbot, placing.R, off)
    assert placing.g_state == "locked"
    click(qtbot, placing.R, off)
    x4, u4 = placing.pairs[3]
    assert np.allclose(u4, off, atol=1)
    assert np.linalg.norm(main.apply(placing.A, [x4])[0] - u4) > 2      # its own placement, not the fit's


def test_locked_ghost_adjustments(placing, dataset, true_A, qtbot):
    x = tissue_points(dataset, 1)[0]
    u = go_to(placing, qtbot, x, true_A)
    click(qtbot, placing.R, u)
    A0 = placing.A_lock.copy()
    vp = placing.R.glw.viewport()
    upp = float(np.mean(placing.R.vb.viewPixelSize()))
    key(qtbot, vp, Qt.Key.Key_Right)
    shift = main.apply(placing.A_lock, [x])[0] - main.apply(A0, [x])[0]
    assert np.isclose(np.linalg.norm(shift), upp, rtol=1e-6)
    key(qtbot, vp, Qt.Key.Key_BracketRight)
    assert np.isclose(placing.g_theta, math.radians(0.5))
    key(qtbot, vp, Qt.Key.Key_Equal)
    assert np.isclose(placing.g_scale, 1.005)
    assert np.isclose(math.sqrt(abs(np.linalg.det(placing.A_lock[:2, :2]))),
                      math.sqrt(abs(np.linalg.det(A0[:2, :2]))) * 1.005)
    key(qtbot, vp, Qt.Key.Key_Escape)
    assert placing.g_state == "float"


def test_overlay_off_hides_ghost(placing, dataset, true_A, qtbot):
    go_to(placing, qtbot, tissue_points(dataset, 1)[0], true_A)
    assert placing.R.ghost.isVisible()
    key(qtbot, placing, Qt.Key.Key_O)
    assert placing.g_state == "off" and not placing.R.ghost.isVisible()


def test_ghost_waits_until_xenium_is_zoomed_in(placing, qtbot, messages):
    set_view(placing.L, 1500, 1200, half=3000)                           # nuclei < MIN_NUC_PX on screen
    settle(qtbot)
    assert not placing._ghost_ok
    assert "Zoom the Xenium pane in" in placing.statusBar().currentMessage()
