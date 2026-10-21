"""Shared fixtures. GUI tests run headless (QT_QPA_PLATFORM=offscreen) through pytest-qt."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")       # before Qt is imported

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

import dummy_data
import main


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """The synthetic Xenium-like dataset (generated once per test run). See tests/dummy_data.py."""
    return dummy_data.generate(str(tmp_path_factory.mktemp("dummy")))


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Never touch the user's real st_align settings (last-used files)."""
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(main, "app_settings", lambda: QtCore.QSettings(path, QtCore.QSettings.Format.IniFormat))


@pytest.fixture(autouse=True)
def messages(monkeypatch):
    """Record message boxes instead of blocking on them: list of (kind, title, text)."""
    seen = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind,
                            lambda parent, title, text, *a, _k=kind: seen.append((_k, title, text)))
    return seen


@pytest.fixture
def window(qtbot, dataset):
    """Aligner window on the dummy dataset, shown (offscreen) at a realistic size."""
    w = main.MainWindow(dataset["xenium"], dataset["if_path"], dataset["boundaries"])
    qtbot.addWidget(w)
    w.resize(1600, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.wait(50)
    return w


@pytest.fixture
def true_A(dataset):
    return np.array(dataset["A"])
