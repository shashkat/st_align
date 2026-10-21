"""Display: contrast / gamma controls, nucleus outline drawing rules, mirrored cursor."""
import numpy as np
import pytest

import main
from helpers import Qt, key, move, set_view, settle, tissue_points

pytestmark = pytest.mark.gui


def test_render_applies_levels_and_gamma(window):
    p = window.L
    p.set_levels(100, 1100)
    p.gamma_slider.setValue(50)                                          # gamma = GAMMA_MAX ** 0.5
    assert np.isclose(p.gamma, main.GAMMA_MAX ** 0.5)
    v = np.array([[0, 100, 600, 1100, 5000]], np.uint16)
    expected = (np.clip((v.astype(float) - 100) / 1000, 0, 1) ** p.gamma * 255).astype(np.uint8)
    assert np.array_equal(p._render(v), expected)
    assert np.array_equal(p.ov_item.image, p._render(p._ov_raw))         # overview re-rendered


def test_rgb_if_has_contrast_controls(window):
    r = window.R
    assert r.img.rgb and len(r.hcurves) == 3                             # one histogram curve per channel
    r.set_levels(20, 120)
    assert r.ov_item.image.dtype == np.uint8 and r.ov_item.image.max() == 255


def test_region_drag_sets_levels(window):
    window.L.region.setRegion((50, 700))
    assert np.allclose(window.L.levels, (50, 700))


def test_auto_contrast_uses_pixels_on_screen(window, dataset, qtbot):
    x = tissue_points(dataset, 1)[0]
    set_view(window.L, *x, half=200)                                     # full-res tile on screen
    settle(qtbot, 300)
    assert window.L.tile_item.isVisible()
    window.L.auto_contrast()
    assert np.allclose(window.L.levels, window.L._percentile_levels(window.L._tile_raw))
    assert window.L.gamma == 1


def test_channel_switch_auto_contrasts_new_channel(window):
    """Regression: narrowing the histogram bounds used to clamp and overwrite the new levels."""
    p = window.L
    p.set_levels(5, 60000)
    p.chan.setCurrentIndex(1)
    expected = p._percentile_levels(p._ov_raw)
    assert np.allclose(p.levels, expected)
    assert np.allclose(p.region.getRegion(), expected)


def test_outlines_only_when_nuclei_are_legible(window, dataset, qtbot, monkeypatch):
    calls = []
    real = window.bounds.select
    monkeypatch.setattr(window.bounds, "select", lambda *a: (calls.append(a), real(*a))[1])
    window.a_nuc.setChecked(True)
    window.L.vb.autoRange()                                              # whole image: nuclei ~2 px
    settle(qtbot)
    assert window._nuc_screen_px(window.L, window.L.O) < main.MIN_NUC_PX
    assert window.L.outline.xData is None or len(window.L.outline.xData) == 0
    assert calls == [] and "zoom in to see nuclei" in window.statusBar().currentMessage()

    x = tissue_points(dataset, 1)[0]
    set_view(window.L, *x, half=300)
    settle(qtbot)
    assert len(window.L.outline.xData) > 0 and len(calls) == 1

    window.L.vb.translateBy(x=30)                                        # small pan: reuse padded selection
    settle(qtbot)
    assert len(calls) == 1
    window.L.vb.translateBy(x=400)                                       # leaves the padded box
    settle(qtbot)
    assert len(calls) == 2


def test_if_outlines_follow_fit_instantly(window, dataset, true_A, qtbot):
    for x in tissue_points(dataset, 3):
        window._add_pair(x, main.apply(true_A, [x])[0])
    settle(qtbot)
    M = main.qtransform(window.R.O @ window.A)
    assert window.R.outline.transform() == M


def test_mirror_cursor_toggle(window, dataset, true_A, qtbot):
    for x in tissue_points(dataset, 3):
        window._add_pair(x, main.apply(true_A, [x])[0])
    x = tissue_points(dataset, 1)[0]
    set_view(window.L, *x, half=300)
    settle(qtbot)
    move(qtbot, window.L, x)
    assert len(window.R.cursor.data) == 1
    key(qtbot, window, Qt.Key.Key_M)
    assert not window.a_cur.isChecked() and len(window.R.cursor.data) == 0
    move(qtbot, window.L, x + 20)
    assert len(window.R.cursor.data) == 0
