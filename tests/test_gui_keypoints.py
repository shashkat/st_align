"""Keypoints in the aligner window: fitting, residuals, save/load, nudging, selection, list navigation."""
import json
import os

import numpy as np
import pytest

import main
from helpers import Qt, click, double_click_list_row, key, set_view, settle, tissue_points, view_center

pytestmark = pytest.mark.gui


def add_true_pairs(w, dataset, A, n=3):
    for x in tissue_points(dataset, n):
        w._add_pair(x, main.apply(A, [x])[0])


def test_three_pairs_fit_the_true_transform_and_link(window, dataset, true_A):
    add_true_pairs(window, dataset, true_A)
    assert np.allclose(window.A, true_A, atol=1e-6)
    assert np.allclose(window._res_um, 0, atol=1e-6)
    assert "exactly 3 pairs" in window.rms.text()
    assert window.a_link.isChecked() and window.a_nuc.isChecked()       # first fit links + shows nuclei


def test_residuals_flag_the_bad_keypoint(window, dataset, true_A):
    add_true_pairs(window, dataset, true_A, n=5)
    a, b = window.pairs[3]
    window.pairs[3] = (a, b + [20.0, 0.0])                               # 20 IF px off
    window._fit()
    assert int(np.argmax(window._res_um)) == 3 and window._res_um[3] > 1.0


def test_collinear_points_are_degenerate(window):
    for t in (0, 1, 2):
        window._add_pair(np.array([100.0 * t, 100.0 * t]), np.array([50.0 * t, 50.0 * t]))
    assert window.A is None and "Degenerate" in window.rms.text()


def test_save_writes_xenium_explorer_format(window, dataset, true_A, tmp_path):
    add_true_pairs(window, dataset, true_A, n=4)
    window.save_to(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["alignment.json", "if_px_to_xenium_px.csv", "keypoints.csv",
                                            "xenium_px_to_if_px.csv"]
    lines = (tmp_path / "keypoints.csv").read_text().splitlines()
    assert lines[0] == "fixedX,fixedY,alignmentX,alignmentY" and len(lines) == 5
    meta = json.loads((tmp_path / "alignment.json").read_text())
    assert np.allclose(meta["xenium_px_to_if_px"], true_A, atol=1e-6)
    assert np.allclose(np.loadtxt(tmp_path / "if_px_to_xenium_px.csv", delimiter=","), np.linalg.inv(true_A))
    assert len(meta["keypoint_residuals_um"]) == 4


def test_keypoints_round_trip_byte_identical(window, tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("fixedX,fixedY,alignmentX,alignmentY\n"
                   "11078.695050801583,18083.55367342278,8000.443834901447,885.4848635814894\n"
                   "16962.510613396826,4198.982038574214,828.9482758410009,3930.0416975256203\n"
                   "11912.84560140568,11488.163138829792,4589.999361424072,1317.5043519174476\n")
    window.load_keypoints(str(src))
    out = tmp_path / "out"
    out.mkdir()
    window.save_to(str(out))
    assert (out / "keypoints.csv").read_bytes() == src.read_bytes()


def test_load_accepts_bom_spaces_reordered_and_legacy(window, tmp_path, messages):
    f = tmp_path / "a.csv"
    f.write_text("﻿ alignmentX, alignmentY ,fixedX,fixedY\n3,4,1,2\n7,8,5,6\n\n", encoding="utf-8")
    window.load_keypoints(str(f))
    assert [(p[0].tolist(), p[1].tolist()) for p in window.pairs] == [([1, 2], [3, 4]), ([5, 6], [7, 8])]
    f.write_text("xenium_x_px,xenium_y_px,if_x_px,if_y_px,xenium_x_um,xenium_y_um,residual_um\n1,2,3,4,0,0,0\n")
    window.load_keypoints(str(f))
    assert [(p[0].tolist(), p[1].tolist()) for p in window.pairs] == [([1, 2], [3, 4])]
    assert messages == []


def test_load_rejects_unknown_columns(window, tmp_path, messages):
    window._add_pair([1, 2], [3, 4])
    f = tmp_path / "bad.csv"
    f.write_text("x,y,u,v\n1,2,3,4\n")
    window.load_keypoints(str(f))
    assert len(window.pairs) == 1                                        # unchanged
    assert messages and messages[0][0] == "warning" and "fixedX" in messages[0][2]


def test_arrow_keys_nudge_selected_keypoint_and_overlay(window, dataset, true_A, qtbot):
    add_true_pairs(window, dataset, true_A)
    window.list.setCurrentRow(1)
    assert window.sel == 1
    before, t0 = window.pairs[1][1].copy(), window.R.outline.transform()
    upp = float(np.mean(window.R.vb.viewPixelSize()))
    window.R.glw.viewport().setFocus()                                   # arrows must reach the shortcut
    key(qtbot, window.R.glw.viewport(), Qt.Key.Key_Right)
    key(qtbot, window.R.glw.viewport(), Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    expected = np.linalg.inv(window.R.O)[:2, :2] @ [upp, 10 * upp]
    assert np.allclose(window.pairs[1][1] - before, expected)
    assert window.R.outline.transform() != t0                            # overlay re-mapped immediately


def test_click_selects_keypoint_when_not_placing(window, dataset, true_A, qtbot):
    add_true_pairs(window, dataset, true_A)
    window.a_link.setChecked(False)
    x, u = window.pairs[2]
    set_view(window.R, *u, half=400)
    settle(qtbot)
    click(qtbot, window.R, u)
    assert window.sel == 2 and len(window.pairs) == 3


@pytest.mark.parametrize("linked", [True, False])
def test_double_click_in_list_jumps_both_panes(window, dataset, true_A, qtbot, linked):
    """Regression: selecting a row used to rebuild the list and swallow the double-click."""
    add_true_pairs(window, dataset, true_A)
    window.a_link.setChecked(linked)
    window.sel = 0
    window._redraw_keypoints()
    set_view(window.L, 50, 50, half=200)
    set_view(window.R, 50, 50, half=100)
    double_click_list_row(qtbot, window.list, 2)
    x, u = window.pairs[2]
    assert window.sel == 2
    assert np.allclose(view_center(window.L), x, atol=1)
    assert np.allclose(view_center(window.R), u, atol=1)


def test_undo_and_delete(window, dataset, true_A):
    add_true_pairs(window, dataset, true_A, n=4)
    window._undo()
    assert len(window.pairs) == 3
    first = window.pairs[0][0].copy()
    window.list.setCurrentRow(0)
    window._delete_selected()
    assert len(window.pairs) == 2 and not np.allclose(window.pairs[0][0], first)
    assert window.A is None and not window.a_link.isChecked()            # < 3 pairs: unfitted, unlinked
