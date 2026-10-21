"""Auto-align from the aligner window: options dialog, placing keypoints, mirroring, failures."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

import autoalign
import main
from helpers import ORIENT_DUMMY

pytestmark = pytest.mark.gui


def test_run_places_keypoints_and_fits(window, true_A, messages):
    window.R.set_orientation(ORIENT_DUMMY)
    r = window.run_auto_align("gray", n_keypoints=8)
    assert r is not None and not r["mirrored"]
    assert len(window.pairs) == 8
    assert np.allclose(window.A[:2, :2], true_A[:2, :2], atol=0.005)
    assert np.linalg.norm(main.apply(window.A, [[1500, 1200]]) - main.apply(true_A, [[1500, 1200]])) < 1
    assert window.a_link.isChecked() and window.a_nuc.isChecked()
    assert messages[-1][0] == "information" and "Placed 8 keypoints" in messages[-1][2]


def test_mirrored_display_is_flipped_to_match(window, messages):
    assert np.allclose(window.R.O, np.eye(3))                            # dummy IF needs a mirror
    window.run_auto_align("gray", n_keypoints=4)
    assert np.isclose(np.linalg.det(window.R.O[:2, :2]), -1)             # pane now flipped
    assert "flipped" in messages[-1][2]


def test_replace_or_keep_existing_keypoints(window):
    window.R.set_orientation(ORIENT_DUMMY)
    window._add_pair([10, 10], [20, 20])
    window.run_auto_align("gray", n_keypoints=4, replace=False)
    assert len(window.pairs) == 5 and np.allclose(window.pairs[0][0], [10, 10])
    window.run_auto_align("gray", n_keypoints=4, replace=True)
    assert len(window.pairs) == 4


def test_failure_leaves_keypoints_unchanged(window, monkeypatch, messages):
    window._add_pair([10, 10], [20, 20])

    def fail(*a, **k):
        raise autoalign.AutoAlignError("no luck")
    monkeypatch.setattr(autoalign, "auto_keypoints", fail)
    assert window.run_auto_align() is None
    assert len(window.pairs) == 1
    assert messages == [("warning", "Automatic alignment failed", "no luck")]


def test_options_dialog_passes_choices(window, qtbot, monkeypatch):
    got = []
    monkeypatch.setattr(window, "run_auto_align", lambda *a, **k: got.append((a, k)))
    window._add_pair([10, 10], [20, 20])
    seen = {}

    def drive():
        dlg = next((d for d in QtWidgets.QApplication.topLevelWidgets()
                    if isinstance(d, QtWidgets.QDialog) and d.isVisible()
                    and d.windowTitle() == "Automatic alignment"), None)
        if dlg is None:
            QtCore.QTimer.singleShot(20, drive)
            return
        combo = dlg.findChild(QtWidgets.QComboBox)
        seen["modes"] = [combo.itemData(i) for i in range(combo.count())]
        combo.setCurrentIndex(seen["modes"].index("hematoxylin"))
        dlg.findChild(QtWidgets.QSpinBox).setValue(7)
        dlg.findChild(QtWidgets.QCheckBox).setChecked(False)              # keep the existing keypoint
        dlg.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    QtCore.QTimer.singleShot(0, drive)
    window._auto_align()
    assert seen["modes"] == list(autoalign.IF_MODES)                     # RGB IF: all channel options
    assert got == [(("hematoxylin", 7), {"replace": False})]
