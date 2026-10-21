"""Start window (file selection) and the start window <-> aligner session cycle."""
import argparse
import os

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

import main

pytestmark = pytest.mark.gui


def make_args(**kw):
    a = dict(xenium=None, if_path=None, boundaries=None, keypoints=None, pixel_size=None)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.fixture
def dialog(qtbot):
    def make(initial=None):
        d = main.StartDialog(initial)
        qtbot.addWidget(d)
        return d
    return make


def test_start_needs_both_images(dialog, dataset):
    d = dialog()
    assert not d.start.isEnabled() and "required" in d.msg.text()
    d.edits["xenium"].setText(dataset["xenium"])
    d.edits["if_path"].setText(dataset["if_path"] + ".missing")
    assert not d.start.isEnabled() and "file not found" in d.msg.text()
    d.edits["if_path"].setText(dataset["if_path"])
    assert d.start.isEnabled()


def test_boundaries_filled_from_xenium_output_layout(dialog, dataset):
    d = dialog()
    d.edits["xenium"].setText(dataset["xenium"])                         # .../morphology_focus/ch0000...
    assert d.edits["boundaries"].text() == dataset["boundaries"]
    d = dialog({"xenium": dataset["xenium"]})                            # also when prefilled at start
    assert d.edits["boundaries"].text() == dataset["boundaries"]
    d = dialog({"xenium": dataset["xenium"], "boundaries": "/somewhere/cell_boundaries.parquet"})
    assert d.edits["boundaries"].text() == "/somewhere/cell_boundaries.parquet"   # explicit choice kept


def test_no_boundaries_is_a_warning_not_an_error(dialog, dataset):
    d = dialog({"xenium": dataset["xenium"], "if_path": dataset["if_path"]})
    d.edits["boundaries"].setText("")
    assert d.start.isEnabled() and "unavailable" in d.msg.text()


def test_last_files_are_remembered(dialog, dataset):
    d = dialog({"xenium": dataset["xenium"], "if_path": dataset["if_path"]})
    d.remember()
    d2 = dialog()
    assert d2.values()["xenium"] == dataset["xenium"] and d2.values()["if_path"] == dataset["if_path"]


@pytest.fixture
def session(qtbot, monkeypatch):
    quits = []
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda: quits.append(1))
    made = []

    def make(**kw):
        s = main.Session(make_args(**kw))
        s.quits = quits
        qtbot.addWidget(s.dlg)
        made.append(s)
        return s
    yield make
    for s in made:                                                       # don't leak windows into other tests
        QtWidgets.QApplication.instance().removeEventFilter(s)
        if s.w is not None:
            s.w.close()


def test_cli_starts_aligner_and_closing_returns_to_start_window(session, dataset, qtbot):
    s = session(xenium=dataset["xenium"], if_path=dataset["if_path"], boundaries=dataset["boundaries"])
    s.start()
    first = s.w
    assert first.isVisible() and not s.dlg.isVisible()
    destroyed = []
    first.destroyed.connect(lambda: destroyed.append(1))
    first.close()
    qtbot.waitUntil(lambda: s.dlg.isVisible())
    qtbot.waitUntil(lambda: destroyed == [1])                             # the closed aligner is freed
    assert s.dlg.values()["xenium"] == dataset["xenium"]                 # prefilled with what was used
    s.dlg.start.click()
    assert s.w is not None and s.w.isVisible() and not s.dlg.isVisible()


def test_quitting_start_window_quits_app(session):
    s = session()
    s.start()
    assert s.dlg.isVisible()
    s.dlg.reject()
    assert s.quits == [1]


def test_app_quit_does_not_reopen_start_window(session, dataset, qtbot):
    s = session(xenium=dataset["xenium"], if_path=dataset["if_path"])
    s.start()
    s.eventFilter(QtWidgets.QApplication.instance(), QtCore.QEvent(QtCore.QEvent.Type.Quit))
    s.w.close()
    qtbot.wait(50)
    assert not s.dlg.isVisible()


def test_unreadable_image_reports_and_returns_to_start_window(session, dataset, messages):
    s = session(xenium=dataset["xenium"], if_path=dataset["boundaries"])   # a parquet is not a TIFF
    s.start()
    assert s.w is None and s.dlg.isVisible()
    assert messages and messages[0][0] == "critical"


def test_paths_remembered_only_after_successful_start(session, dataset):
    s = session()
    s.start()
    s.dlg.edits["xenium"].setText(dataset["xenium"])
    s.dlg.edits["if_path"].setText(dataset["boundaries"])                # exists, but unreadable
    s.dlg.start.click()
    assert main.app_settings().value("if_path", "", str) == ""
    s.dlg.edits["if_path"].setText(dataset["if_path"])
    s.dlg.start.click()
    assert os.path.samefile(main.app_settings().value("if_path", "", str), dataset["if_path"])
