#!/usr/bin/env python3
"""
st_align — two-pane prototype for aligning a post-Xenium IF image to the Xenium DAPI image.

Left pane  : Xenium morphology image (fixed), e.g. morphology_focus/morphology_focus_0000.ome.tif
Right pane : post-Xenium IF image (moving), any OME-TIFF / TIFF (pyramidal strongly recommended)

Workflow (overlay placement, needs --boundaries)
  1. Flip / rotate the IF pane until it looks like the DAPI pane (display only; the raw IF
     pixel coordinates are what get exported, so you never have to flip the image in code).
  2. Zoom both panes to roughly the same scale, then press O (Overlay place). The Xenium nuclei
     visible in the left pane now float over the IF pane, following the mouse (the left pane's
     centre sits under the cursor). Wheel-zoom the IF pane to fine-tune scale; [ / ] rotate the
     ghost, - / = scale it.
  3. Click when the outlines match the IF nuclei: the overlay locks in place. Arrow keys nudge the
     locked overlay (Shift = 10x), Esc unlocks. Then click the spot you are most confident about:
     that becomes the keypoint. Pan the Xenium pane elsewhere and repeat for keypoints 2 and 3.
  4. After 3 keypoints an affine is fitted, the views are linked and the nuclei overlay (red) is
     drawn through the fit. Overlay mode keeps working the same way: float, lock and click to add
     keypoint 4, 5, ... (with linked views, use - / = rather than the wheel to adjust the scale).
     To refine an existing keypoint, pick it in the list (or turn overlay mode off and click it) and
     use the arrow keys to nudge its IF position: the red overlay updates immediately.
  5. Save (Ctrl+S) writes keypoints + both affine directions to a folder of your choice.
     keypoints.csv uses Xenium Explorer's format (fixedX,fixedY,alignmentX,alignmentY: Xenium px,
     raw IF px), so it can be imported there, and Xenium Explorer exports can be loaded here.

Classic placement: press K and click matching points, one click in each pane = one pair.

Keys:  O overlay place · K pair-click mode · L link · N nuclei · M mirror cursor · arrows nudge (Shift x10)
       [ ] rotate ghost · - = scale ghost · Esc unlock · Ctrl+Z undo · Del delete selected · Ctrl+S save
       Select a keypoint in the list (or click it with overlay mode off); double-click to jump to it.

Run:
  python main.py                 # opens a start window to pick the files
  python main.py --xenium morphology_focus/ch0000_dapi.ome.tif --if post_if.ome.tif \\
                 --boundaries nucleus_boundaries.parquet      # starts the aligner directly
  (Giving only some of the paths opens the start window with those filled in.)
  Closing the aligner window returns to the start window; quitting the start window quits.
"""
import argparse
import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import zarr
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

MAX_TILE = 2048          # max tile side (pixels) read for the detail layer
MAX_OVERVIEW = 2048      # max side of the always-visible low-res overview
MAX_CELLS = 25000        # max nucleus outlines drawn at once (backstop; MIN_NUC_PX normally decides)
MIN_NUC_PX = 6           # draw outlines only once a typical nucleus is this many screen px across
OUTLINE_PAD = 0.25       # select outlines for the view padded by this fraction, reuse until it leaves
XENIUM_PX_UM = 0.2125    # Xenium morphology level-0 pixel size (µm)
# keypoints.csv columns, as exported/imported by Xenium Explorer's image alignment:
# fixed = Xenium morphology level-0 px, alignment = raw level-0 px of the aligned (IF / H&E) image
KP_COLS = ("fixedX", "fixedY", "alignmentX", "alignmentY")
KP_COLS_OLD = ("xenium_x_px", "xenium_y_px", "if_x_px", "if_y_px")   # files saved by earlier versions of st_align
GAMMA_MAX = 5.0          # gamma slider spans 1/GAMMA_MAX .. GAMMA_MAX


# ----------------------------------------------------------------------------- geometry helpers
def qtransform(M):
    """3x3 matrix acting on column vectors [x, y, 1] -> QTransform (row-vector convention)."""
    return QtGui.QTransform(M[0, 0], M[1, 0], 0.0,
                            M[0, 1], M[1, 1], 0.0,
                            M[0, 2], M[1, 2], 1.0)


def apply(M, pts):
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return pts @ M[:2, :2].T + M[:2, 2]


def T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def S(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)


def Rot(theta):
    """Rotation by theta radians; positive = clockwise on screen (y points down)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


FLIP_H = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
FLIP_V = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
ROT_CW = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)   # clockwise on screen (y points down)


def bbox(pts):
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def rect_corners(r):
    return np.array([[r.left(), r.top()], [r.right(), r.top()],
                     [r.right(), r.bottom()], [r.left(), r.bottom()]])


# ----------------------------------------------------------------------------- image access
class PyramidImage:
    """Lazy access to a (possibly pyramidal) TIFF via tifffile's zarr store."""

    def __init__(self, path):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        series = self.tif.series[0]
        self.axes = series.axes
        z = zarr.open(series.aszarr(), mode="r")
        if isinstance(z, zarr.Array):
            self.arrays = [z]
        else:
            self.arrays = [z[k] for k in sorted(z.array_keys(), key=int)]
        ax = {a: i for i, a in enumerate(self.axes)}
        if "Y" not in ax or "X" not in ax:
            raise ValueError(f"{path}: need Y and X axes, got {self.axes}")
        if ax["Y"] > ax["X"]:
            raise ValueError(f"{path}: X-before-Y axis order ({self.axes}) not supported")
        self.ax = ax
        shp = self.arrays[0].shape
        self.rgb = "S" in ax and shp[ax["S"]] in (3, 4)
        self.n_channels = shp[ax["C"]] if "C" in ax else 1
        self.shapes = [(a.shape[ax["Y"]], a.shape[ax["X"]]) for a in self.arrays]
        H0, W0 = self.shapes[0]
        self.ds = [(W0 / w, H0 / h) for h, w in self.shapes]          # (dx, dy) per level
        self.channel = 0
        self.channel_names, self.pixel_size = self._ome_info()

    def _ome_info(self):
        names, px = [], None
        try:
            root = ET.fromstring(self.tif.ome_metadata)
            for el in root.iter():
                tag = el.tag.split("}")[-1]
                if tag == "Pixels" and px is None and el.get("PhysicalSizeX"):
                    px = float(el.get("PhysicalSizeX"))
                if tag == "Channel":
                    names.append(el.get("Name") or el.get("ID") or "")
        except Exception:
            pass
        names = names[: self.n_channels] if len(names) >= self.n_channels else []
        names = names or [f"ch {i}" for i in range(self.n_channels)]
        return names, px

    def read(self, level, r0, r1, c0, c1, step=1):
        arr = self.arrays[level]
        idx = []
        for a in self.axes:
            if a == "Y":
                idx.append(slice(r0, r1, step))
            elif a == "X":
                idx.append(slice(c0, c1, step))
            elif a == "C":
                idx.append(self.channel)
            elif a == "S":
                idx.append(slice(None))
            else:
                idx.append(0)
        out = np.asarray(arr[tuple(idx)])
        if "S" in self.ax:
            out = out[..., :3] if self.rgb else out.mean(-1)
        return out

    def overview(self):
        """Low-res whole image + its (dx, dy) relative to level 0."""
        lvl = len(self.arrays) - 1
        h, w = self.shapes[lvl]
        step = max(1, math.ceil(max(h, w) / MAX_OVERVIEW))
        data = self.read(lvl, 0, h, 0, w, step)
        dx, dy = self.ds[lvl]
        return data, (dx * step, dy * step)


# ----------------------------------------------------------------------------- nucleus outlines
class Boundaries:
    """Xenium nucleus_boundaries.parquet (or cell_boundaries.parquet), in level-0 pixels."""

    def __init__(self, path, px_um):
        import pandas as pd
        df = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
        codes, _ = pd.factorize(df["cell_id"], sort=False)
        order = np.argsort(codes, kind="stable")             # each cell's vertices contiguous
        codes = codes[order]
        self.x = df["vertex_x"].to_numpy(float)[order] / px_um
        self.y = df["vertex_y"].to_numpy(float)[order] / px_um
        n = codes.max() + 1
        self.cnt = np.bincount(codes, minlength=n)
        self.start = np.concatenate([[0], np.cumsum(self.cnt)[:-1]])
        self.cx = np.bincount(codes, weights=self.x, minlength=n) / self.cnt
        self.cy = np.bincount(codes, weights=self.y, minlength=n) / self.cnt
        self.connect = np.ones(len(codes), dtype=bool)       # False on each cell's last vertex
        self.connect[self.start + self.cnt - 1] = False
        w = np.maximum.reduceat(self.x, self.start) - np.minimum.reduceat(self.x, self.start)
        h = np.maximum.reduceat(self.y, self.start) - np.minimum.reduceat(self.y, self.start)
        self.typical_size = float(np.median(np.maximum(w, h)))  # Xenium px

    def select(self, x0, y0, x1, y1):
        """Outlines of the cells whose centroid is in the box; cost scales with the selection."""
        cells = np.flatnonzero((self.cx >= x0) & (self.cx <= x1) & (self.cy >= y0) & (self.cy <= y1))
        n = len(cells)
        if n == 0 or n > MAX_CELLS:
            return None, n
        c = self.cnt[cells]
        idx = np.arange(c.sum()) + np.repeat(self.start[cells] - np.cumsum(c) + c, c)
        return (np.column_stack([self.x[idx], self.y[idx]]), self.connect[idx]), n


# ----------------------------------------------------------------------------- one image pane
class Pane(QtWidgets.QWidget):
    clicked = QtCore.Signal(float, float)        # raw level-0 pixel coords
    hovered = QtCore.Signal(float, float)        # raw level-0 pixel coords
    settled = QtCore.Signal()                    # view stopped changing (debounced)

    def __init__(self, title, img):
        super().__init__()
        self.img = img
        self.O = np.eye(3)                       # orientation: raw px -> scene (display only)
        self._tile_key = None

        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)

        self.ov_item = pg.ImageItem()
        self.tile_item = pg.ImageItem()
        self.tile_item.setZValue(1)
        self.outline = pg.PlotCurveItem(pen=pg.mkPen((255, 70, 70), width=1))
        self.outline.setZValue(5)
        self.ghost = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=1))   # floating overlay
        self.ghost.setZValue(6)
        self.ghost.hide()
        self.kp = pg.ScatterPlotItem(size=12, symbol="o", pen=pg.mkPen((0, 255, 120), width=2),
                                     brush=None)
        self.kp.setZValue(10)
        self.sel_kp = pg.ScatterPlotItem(size=20, symbol="o", pen=pg.mkPen((255, 0, 255), width=2),
                                         brush=None)
        self.sel_kp.setZValue(10)
        self.pending = pg.ScatterPlotItem(size=14, symbol="x", pen=pg.mkPen((255, 255, 0), width=2))
        self.pending.setZValue(11)
        self.cursor = pg.ScatterPlotItem(size=22, symbol="+", pen=pg.mkPen((255, 230, 0), width=2))
        self.cursor.setZValue(12)
        for it in (self.ov_item, self.tile_item, self.outline, self.ghost, self.kp, self.sel_kp,
                   self.pending, self.cursor):
            self.vb.addItem(it)
        self.labels = []
        self.last_move = None                    # last mouse position (graphics-scene coords)
        self.last_click = None

        # display: contrast (min-max) + gamma, applied through a lookup table to overview and tile
        self.levels = (0.0, 1.0)
        self.gamma = 1.0
        self._lut, self._lut_key = None, None
        self._ov_raw = self._tile_raw = None
        self._hist_src = None

        self.hplot = pg.PlotWidget()
        self.hplot.setFixedHeight(40)
        self.hplot.hideAxis("left")
        self.hplot.hideAxis("bottom")
        self.hplot.hideButtons()
        self.hplot.setMenuEnabled(False)
        self.hplot.setMouseEnabled(x=False, y=False)
        self.hplot.setToolTip("Drag the edges to set min / max contrast · double-click for the full range")
        self.hcurves = []
        self.region = pg.LinearRegionItem(orientation="vertical", brush=(255, 255, 255, 45))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._region_changed)
        self.hplot.addItem(self.region)
        self.hplot.scene().sigMouseClicked.connect(
            lambda ev: ev.double() and self.set_levels(*self._hrange))
        self.lv_label = QtWidgets.QLabel("")
        self.lv_label.setStyleSheet("color: gray")
        self.lv_label.setMinimumWidth(90)
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(-100, 100)                   # gamma = GAMMA_MAX ** (v / 100)
        self.gamma_slider.setFixedWidth(110)
        self.gamma_slider.setToolTip("Gamma (<1 brightens dim signal, >1 darkens) · double-click to reset")
        self.gamma_slider.valueChanged.connect(self._gamma_changed)
        self.gamma_slider.mouseDoubleClickEvent = lambda ev: self.gamma_slider.setValue(0)
        self.gamma_label = QtWidgets.QLabel("γ 1.00")
        self.gamma_label.setMinimumWidth(45)
        auto = QtWidgets.QToolButton()
        auto.setText("Auto")
        auto.setToolTip("Auto contrast for the pixels on screen (1st–99.7th percentile) and gamma 1")
        auto.clicked.connect(self.auto_contrast)

        head = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"<b>{title}</b>  <span style='color:gray'>{os.path.basename(img.path)}</span>")
        head.addWidget(lbl, 1)
        self.chan = QtWidgets.QComboBox()
        self.chan.addItems(img.channel_names)
        self.chan.setVisible(img.n_channels > 1)
        self.chan.currentIndexChanged.connect(self.set_channel)
        head.addWidget(self.chan)
        self.info = QtWidgets.QLabel("")
        self.info.setStyleSheet("color: gray")

        disp = QtWidgets.QHBoxLayout()
        disp.addWidget(QtWidgets.QLabel("Contrast"))
        disp.addWidget(self.hplot, 1)
        disp.addWidget(self.lv_label)
        disp.addWidget(self.gamma_slider)
        disp.addWidget(self.gamma_label)
        disp.addWidget(auto)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(head)
        lay.addLayout(disp)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.info)

        self.timer = QtCore.QTimer(self, singleShot=True, interval=80)
        self.timer.timeout.connect(self._settle)
        self.vb.sigRangeChanged.connect(lambda *a: self.timer.start())
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        self.glw.scene().sigMouseMoved.connect(self._on_move)

        self.load_overview(auto_contrast=True)
        self.vb.autoRange(padding=0.02)

    # -- coordinates
    def to_scene(self, raw):
        return apply(self.O, raw)

    def to_raw(self, scene):
        return apply(np.linalg.inv(self.O), scene)

    def visible_raw_bbox(self, pad=0.0):
        r = self.vb.viewRect()
        if pad:
            r = r.adjusted(-r.width() * pad, -r.height() * pad, r.width() * pad, r.height() * pad)
        return bbox(self.to_raw(rect_corners(r)))

    # -- image layers
    def load_overview(self, auto_contrast=False):
        data, (dx, dy) = self.img.overview()
        self._ov_raw = data
        self.ov_item.setTransform(qtransform(self.O @ S(dx, dy)))
        self._ov_ds = max(dx, dy)
        self._set_histogram(data)
        if auto_contrast:
            if self.img.rgb and data.dtype == np.uint8:
                lo, hi = 0.0, 255.0
            else:
                lo, hi = self._percentile_levels(data)
            self.levels = (lo, hi)
        self._show_levels()
        self.ov_item.setImage(self._render(data), levels=(0, 255), autoLevels=False)
        self._tile_key = None

    # -- contrast / gamma
    def _displayed_raw(self):
        return self._tile_raw if self.tile_item.isVisible() and self._tile_raw is not None else self._ov_raw

    def _set_histogram(self, data):
        """Histogram strip of the pixels on screen (pyramid levels differ: the overview is blurred)."""
        if data is self._hist_src:
            return
        self._hist_src = data
        if data.dtype == np.uint8:
            lo0, hi0 = 0.0, 255.0
        else:
            lo0, hi0 = float(data.min()), float(data.max())
        hi0 = max(hi0, lo0 + 1)
        self._hrange = (lo0, hi0)
        step = max(1, int(math.sqrt(data.shape[0] * data.shape[1] / 1e6)))   # ~1 Mpx sample
        sample = data[::step, ::step]
        chans = [sample[..., i] for i in range(3)] if sample.ndim == 3 else [sample]
        colors = [(255, 80, 80), (80, 255, 80), (90, 140, 255)] if len(chans) == 3 else [(200, 200, 200)]
        for c in self.hcurves:
            self.hplot.removeItem(c)
        self.hcurves = []
        for ch, col in zip(chans, colors):
            cnt, edges = np.histogram(ch, bins=256, range=(lo0, hi0))
            y = np.log1p(cnt.astype(float))
            y /= max(y.max(), 1e-9)
            c = pg.PlotCurveItem(edges, y, stepMode="center", fillLevel=0, pen=pg.mkPen(col),
                                 brush=pg.mkBrush(*col, 60))
            self.hplot.addItem(c)
            self.hcurves.append(c)
        self._fit_hrange()

    def _fit_hrange(self):
        """Strip spans the on-screen data range, widened to keep the current levels reachable."""
        lo = min(self._hrange[0], self.levels[0])
        hi = max(self._hrange[1], self.levels[1])
        self.hplot.setXRange(lo, hi, padding=0)
        self.hplot.setYRange(0, 1.05, padding=0)
        self.region.blockSignals(True)          # clamping the handles must not overwrite self.levels
        self.region.setBounds((lo, hi))
        self.region.blockSignals(False)

    @staticmethod
    def _percentile_levels(data):
        nz = data[data > 0]
        lo, hi = np.percentile(nz if nz.size else data, [1, 99.7])
        return float(lo), float(max(hi, lo + 1))

    def _show_levels(self):
        lo, hi = self.levels
        self._fit_hrange()
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)
        self.lv_label.setText(f"{lo:.0f} – {hi:.0f}")

    def _region_changed(self):
        lo, hi = self.region.getRegion()
        self.levels = (float(lo), float(max(hi, lo + 1e-6)))
        self.lv_label.setText(f"{lo:.0f} – {hi:.0f}")
        self._redisplay()

    def _gamma_changed(self, v):
        self.gamma = GAMMA_MAX ** (v / 100)
        self.gamma_label.setText(f"γ {self.gamma:.2f}")
        self._redisplay()

    def set_levels(self, lo, hi):
        self.levels = (float(lo), float(hi))
        self._show_levels()
        self._redisplay()

    def auto_contrast(self):
        self.levels = self._percentile_levels(self._displayed_raw())
        self._show_levels()
        self.gamma_slider.setValue(0)                           # re-renders if gamma changed
        self._redisplay()

    def _render(self, data):
        """Raw pixels -> uint8 display values: clip to levels, normalise, apply gamma."""
        lo, hi = self.levels
        if data.dtype in (np.uint8, np.uint16):                 # lookup table, fast for RGB too
            key = (data.dtype, lo, hi, self.gamma)
            if key != self._lut_key:
                x = np.arange(np.iinfo(data.dtype).max + 1, dtype=np.float32)
                self._lut = (np.clip((x - lo) / (hi - lo), 0, 1) ** self.gamma * 255).astype(np.uint8)
                self._lut_key = key
            return self._lut[data]
        x = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0, 1)
        return (x ** self.gamma * 255).astype(np.uint8)

    def _redisplay(self):
        if self._ov_raw is not None:
            self.ov_item.setImage(self._render(self._ov_raw), levels=(0, 255), autoLevels=False)
        if self._tile_raw is not None and self.tile_item.isVisible():
            self.tile_item.setImage(self._render(self._tile_raw), levels=(0, 255), autoLevels=False)

    def set_channel(self, i):
        self.img.channel = i
        self.load_overview(auto_contrast=True)
        self.refresh_tile()

    def set_orientation(self, O):
        self.O = O
        self.load_overview()
        self.refresh_tile()
        self.vb.autoRange(padding=0.02)

    def refresh_tile(self):
        img = self.img
        x0, y0, x1, y1 = self.visible_raw_bbox(pad=0.15)
        need = self.visible_raw_bbox()
        upp = max(self.vb.viewPixelSize())                      # raw px per screen px
        lvl = 0
        for i, (dx, dy) in enumerate(img.ds):
            if max(dx, dy) <= upp * 1.25:
                lvl = i
        while True:
            dx, dy = img.ds[lvl]
            if (x1 - x0) / dx <= MAX_TILE and (y1 - y0) / dy <= MAX_TILE:
                break
            if lvl == len(img.ds) - 1:
                self.tile_item.hide()
                self.info.setText(f"overview only (1/{self._ov_ds:.0f})")
                return
            lvl += 1
        dx, dy = img.ds[lvl]
        if max(dx, dy) >= self._ov_ds:                          # overview is already this sharp
            self.tile_item.hide()
            self.info.setText(f"overview (1/{self._ov_ds:.0f})")
            return
        h, w = img.shapes[lvl]
        c0, c1 = max(0, int(x0 / dx)), min(w, int(math.ceil(x1 / dx)))
        r0, r1 = max(0, int(y0 / dy)), min(h, int(math.ceil(y1 / dy)))
        if c1 <= c0 or r1 <= r0:
            self.tile_item.hide()
            return
        k = self._tile_key
        if k and k[0] == lvl and k[5] == img.channel and \
                k[1] * dx <= max(0, need[0]) and k[3] * dy <= max(0, need[1]) and \
                k[2] * dx >= min(w * dx, need[2]) and k[4] * dy >= min(h * dy, need[3]):
            return                                              # current tile still covers the view
        data = img.read(lvl, r0, r1, c0, c1)
        self._tile_raw = data
        self.tile_item.setImage(self._render(data), levels=(0, 255), autoLevels=False)
        self.tile_item.setTransform(qtransform(self.O @ T(c0 * dx, r0 * dy) @ S(dx, dy)))
        self.tile_item.show()
        self._set_histogram(data)
        self._tile_key = (lvl, c0, c1, r0, r1, img.channel)
        self.info.setText(f"level {lvl} (1/{max(dx, dy):.0f})")

    def _settle(self):
        self.refresh_tile()
        self._set_histogram(self._displayed_raw())
        self.settled.emit()

    # -- overlays
    def set_keypoints(self, raw_pts, pending_raw=None, sel=None):
        for t in self.labels:
            self.vb.removeItem(t)
        self.labels = []
        if len(raw_pts):
            sp = self.to_scene(raw_pts)
            self.kp.setData(pos=sp)
            for i, p in enumerate(sp):
                t = pg.TextItem(str(i + 1), color=(0, 255, 120), anchor=(-0.3, 1.1))
                t.setPos(*p)
                t.setZValue(10)
                self.vb.addItem(t)
                self.labels.append(t)
        else:
            self.kp.setData(pos=np.empty((0, 2)))
        self.sel_kp.setData(pos=self.to_scene(raw_pts[sel]) if sel is not None else np.empty((0, 2)))
        self.pending.setData(pos=self.to_scene(pending_raw) if pending_raw is not None else np.empty((0, 2)))

    @staticmethod
    def _set_curve(item, res, M):
        """res = (Xenium px vertices, connect) from Boundaries.select; M maps Xenium px -> scene.
        Vertices stay in Xenium px and only the item transform changes, so re-mapping is instant."""
        if res is None:
            item.setData(x=np.array([]), y=np.array([]))
        else:
            item.setData(x=res[0][:, 0], y=res[0][:, 1], connect=res[1])
            item.setTransform(qtransform(M))

    def set_outline(self, res, M=None):
        self._set_curve(self.outline, res, M)

    def set_ghost(self, res):
        self._set_curve(self.ghost, res, np.eye(3))

    def hit(self, raw_pts, scene_pos, tol=10):
        """Index of the keypoint within tol screen px of scene_pos (graphics-scene coords), or None."""
        if not len(raw_pts):
            return None
        d = []
        for p in self.to_scene(raw_pts):
            q = self.vb.mapViewToScene(QtCore.QPointF(*p)) - scene_pos
            d.append(math.hypot(q.x(), q.y()))
        i = int(np.argmin(d))
        return i if d[i] <= tol else None

    def set_cursor(self, raw_pt):
        self.cursor.setData(pos=self.to_scene(raw_pt) if raw_pt is not None else np.empty((0, 2)))

    def center_on(self, raw_pt):
        r = self.vb.viewRect()
        c = self.to_scene(raw_pt)[0]
        self.vb.setRange(QtCore.QRectF(c[0] - r.width() / 2, c[1] - r.height() / 2, r.width(), r.height()),
                         padding=0)

    # -- mouse
    def _on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.last_click = ev.scenePos()
        p = self.vb.mapSceneToView(ev.scenePos())
        raw = self.to_raw([p.x(), p.y()])[0]
        self.clicked.emit(float(raw[0]), float(raw[1]))

    def _on_move(self, pos):
        if not self.vb.sceneBoundingRect().contains(pos):
            return
        self.last_move = pos
        p = self.vb.mapSceneToView(pos)
        raw = self.to_raw([p.x(), p.y()])[0]
        self.hovered.emit(float(raw[0]), float(raw[1]))


# ----------------------------------------------------------------------------- main window
class MainWindow(QtWidgets.QMainWindow):
    closed = QtCore.Signal()

    def __init__(self, xen_path, if_path, boundaries_path=None, px_um=None):
        super().__init__()
        self.setWindowTitle("st_align — Xenium ↔ post-Xenium IF")
        self.xen = PyramidImage(xen_path)
        self.ifim = PyramidImage(if_path)
        self.px_um = px_um or self.xen.pixel_size or XENIUM_PX_UM
        self.bounds = None
        if boundaries_path:
            self.statusBar().showMessage("Loading boundaries…")
            QtWidgets.QApplication.processEvents()
            self.bounds = Boundaries(boundaries_path, self.px_um)

        self.L = Pane("Xenium (fixed)", self.xen)
        self.R = Pane("IF (moving)", self.ifim)
        split = QtWidgets.QSplitter()
        split.addWidget(self.L)
        split.addWidget(self.R)

        self.list = QtWidgets.QListWidget()
        self.list.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)      # keep arrow keys for nudging
        self.list.itemDoubleClicked.connect(self._jump_to_pair)
        self.list.currentRowChanged.connect(self._list_row_changed)
        self.rms = QtWidgets.QLabel("Need 3 pairs to fit")
        self.rms.setWordWrap(True)
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addWidget(QtWidgets.QLabel("<b>Keypoint pairs</b> (residual in Xenium µm)"))
        sl.addWidget(self.list, 1)
        sl.addWidget(self.rms)
        dock = QtWidgets.QDockWidget("Keypoints")
        dock.setWidget(side)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.setCentralWidget(split)

        # state
        self.pairs = []                 # list of (xen_raw[2], if_raw[2])
        self.pending = {"L": None, "R": None}
        self.A = None                   # 3x3: Xenium px -> IF raw px
        self.sel = None                 # selected pair index
        self._syncing = False
        self._active = "L"
        # overlay placement: "float" (ghost follows mouse), "locked" (ghost pinned), "off"
        self.g_state = "off"
        self.g_theta = 0.0              # extra ghost rotation (rad), kept between placements
        self.g_scale = 1.0              # extra ghost scale on top of the screen-zoom ratio
        self.A_lock = None              # 3x3: Xenium px -> IF raw px while locked
        self.lock_raw = None            # IF raw px where the ghost was locked (pivot for rotate/scale)
        self._ghost_ok = False          # ghost has outlines to show

        tb = self.addToolBar("main")
        tb.setMovable(False)

        def act(text, slot, key=None, checkable=False, tip=None):
            a = tb.addAction(text)
            a.setCheckable(checkable)
            if key:
                a.setShortcut(QtGui.QKeySequence(key))
            a.setToolTip(f"{tip or text}" + (f"  [{key}]" if key else ""))
            (a.toggled if checkable else a.triggered).connect(slot)
            return a

        act("IF: flip ↔", lambda: self._orient(FLIP_H))
        act("IF: flip ↕", lambda: self._orient(FLIP_V))
        act("IF: rotate 90°", lambda: self._orient(ROT_CW))
        tb.addSeparator()
        self.a_ov = act("Overlay place", self._overlay_toggled, "O", True,
                        "Float the Xenium nuclei over the IF image to place keypoints")
        self.a_ov.setEnabled(self.bounds is not None)
        self.a_kp = act("Pair clicks", self._mode_changed, "K", True,
                        "Click a matching point in each pane")
        act("Undo", self._undo, "Ctrl+Z")
        act("Delete selected", self._delete_selected, "Delete")
        act("Clear", self._clear)
        tb.addSeparator()
        self.a_link = act("Link views", self._link_toggled, "L", True)
        self.a_link.setEnabled(False)
        self.a_nuc = act("Show nuclei", self._update_outlines, "N", True)
        self.a_nuc.setEnabled(self.bounds is not None)
        self.a_cur = act("Mirror cursor", self._mirror_toggled, "M", True,
                         "Show a yellow cross in the other pane where the cursor maps")
        self.a_cur.setChecked(True)
        tb.addSeparator()
        act("Load keypoints…", self._load)
        act("Save…", self._save, "Ctrl+S")

        def key(seq, slot):
            QtGui.QShortcut(QtGui.QKeySequence(seq), self, activated=slot)

        for name, (dx, dy) in {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}.items():
            key(name, lambda dx=dx, dy=dy: self._nudge(dx, dy))
            key("Shift+" + name, lambda dx=dx, dy=dy: self._nudge(10 * dx, 10 * dy))
        key("[", lambda: self._ghost_adjust(rot=-math.radians(0.5)))
        key("]", lambda: self._ghost_adjust(rot=math.radians(0.5)))
        key("-", lambda: self._ghost_adjust(scale=1 / 1.005))
        key("=", lambda: self._ghost_adjust(scale=1.005))
        key("Escape", self._escape)

        self._out_box = {"L": None, "R": None}   # Xenium-px box each pane's outlines were selected for
        self.out_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self.out_timer.timeout.connect(self._update_outlines)

        for side_, pane in (("L", self.L), ("R", self.R)):
            pane.clicked.connect(lambda x, y, s=side_: self._clicked(s, x, y))
            pane.hovered.connect(lambda x, y, s=side_: self._hovered(s, x, y))
            pane.settled.connect(lambda s=side_: self._update_outlines(sides=s))
            pane.vb.sigRangeChanged.connect(lambda *a, s=side_: self._range_changed(s))
            pane.vb.sigRangeChanged.connect(lambda *a: self._update_ghost())
        self.L.settled.connect(self._reload_ghost)

        self.resize(1700, 950)
        self.statusBar().showMessage(
            "Orient the IF image, zoom both panes to a similar scale, then press O (overlay place)."
            if self.bounds is not None else
            "Orient the IF image, then press K and click matching points (left & right).")

    # ---------------------------------------------------------------- transforms
    def _scene_map(self, src):
        """3x3 mapping src-pane scene coords -> other-pane scene coords (needs self.A)."""
        M = self.R.O @ self.A @ np.linalg.inv(self.L.O)     # L scene -> R scene
        return M if src == "L" else np.linalg.inv(M)

    def _current_map(self):
        """Xenium px -> IF raw px mapping for the cursor: the ghost while placing, else the fit."""
        if self.g_state == "locked":
            return self.A_lock
        if self.g_state == "float":
            G = self._float_G()
            if G is not None:
                return np.linalg.inv(self.R.O) @ G
        return self.A

    def _fit(self):
        was_fitted = self.A is not None
        if len(self.pairs) < 3:
            self.A = None
            self.a_link.setChecked(False)
            self.a_link.setEnabled(False)
            self.rms.setText(f"{len(self.pairs)} pair(s) — need 3 to fit")
            return
        X = np.array([p[0] for p in self.pairs])
        U = np.array([p[1] for p in self.pairs])
        Xh = np.column_stack([X, np.ones(len(X))])
        P, *_ = np.linalg.lstsq(Xh, U, rcond=None)           # (3, 2)
        A = np.eye(3)
        A[0, :] = P[:, 0]
        A[1, :] = P[:, 1]
        if abs(np.linalg.det(A[:2, :2])) < 1e-12:
            self.A = None
            self.a_link.setChecked(False)
            self.a_link.setEnabled(False)
            self.rms.setText("Degenerate fit (points collinear?) — spread them out")
            return
        self.A = A
        self.a_link.setEnabled(True)
        back = apply(np.linalg.inv(A), U)                     # IF -> Xenium
        self._res_um = np.linalg.norm(back - X, axis=1) * self.px_um
        lin = A[:2, :2]
        sx, sy = np.linalg.norm(lin[:, 0]), np.linalg.norm(lin[:, 1])
        rot = math.degrees(math.atan2(lin[1, 0], lin[0, 0]))
        rms = float(np.sqrt(np.mean(self._res_um ** 2)))
        note = "" if len(self.pairs) > 3 else " (exactly 3 pairs: residuals are always 0 — add more)"
        self.rms.setText(f"RMS residual: <b>{rms:.2f} µm</b> · max {self._res_um.max():.2f} µm{note}<br>"
                         f"scale x {sx:.4f}, y {sy:.4f} (IF px per Xenium px) · rotation {rot:.2f}°")
        if not was_fitted:                                    # first fit: link views + show overlay
            self.a_link.setChecked(True)
            if self.bounds is not None:
                self.a_nuc.setChecked(True)

    # ---------------------------------------------------------------- keypoints
    def _mode_changed(self, on):
        if on:
            self.a_ov.setChecked(False)
        self.statusBar().showMessage("Pair-click mode: click a point in each pane (any order)." if on
                                     else "Pair-click mode off.")

    def _add_pair(self, xen, ifp):
        self.pairs.append((np.asarray(xen, float), np.asarray(ifp, float)))
        self.sel = len(self.pairs) - 1
        self._pairs_changed()

    def _pairs_changed(self):
        if self.sel is not None and self.sel >= len(self.pairs):
            self.sel = len(self.pairs) - 1 if self.pairs else None
        self._fit()
        self._sync_ghost_state()
        self._redraw_keypoints()

    def _clicked(self, side, x, y):
        pane = self.L if side == "L" else self.R
        if self.a_kp.isChecked():
            self.pending[side] = np.array([x, y])
            if self.pending["L"] is not None and self.pending["R"] is not None:
                self.pending, (a, b) = {"L": None, "R": None}, (self.pending["L"], self.pending["R"])
                self._add_pair(a, b)
            else:
                self._redraw_keypoints()
            return
        if self.a_ov.isChecked() and self.g_state == "float":
            if side == "R":
                self._lock_ghost(x, y)
            return
        if self.a_ov.isChecked() and self.g_state == "locked":
            if side == "R":
                xen = apply(np.linalg.inv(self.A_lock), [[x, y]])[0]
                self.g_state = "float"
                self._add_pair(xen, [x, y])
                n = len(self.pairs)
                self.statusBar().showMessage(
                    f"Keypoint {n} placed. Pan the Xenium pane to a different region and match again "
                    f"({3 - n} more needed)." if self.A is None else
                    f"Keypoint {n} placed{' — views linked' if n == 3 else ''}. Move to another region and "
                    f"match again to add more, or turn overlay mode off (O) to select and nudge keypoints.")
            return
        # selection
        i = pane.hit(np.array([p[0 if side == "L" else 1] for p in self.pairs]).reshape(-1, 2), pane.last_click)
        if i is not None:
            self.sel = i
            self._redraw_keypoints()
            self.statusBar().showMessage(f"Keypoint #{i + 1} selected — arrow keys nudge its IF position "
                                         f"(Shift = 10 px).")

    def _redraw_markers(self):
        L_pts = np.array([p[0] for p in self.pairs]).reshape(-1, 2)
        R_pts = np.array([p[1] for p in self.pairs]).reshape(-1, 2)
        self.L.set_keypoints(L_pts, self.pending["L"], self.sel)
        self.R.set_keypoints(R_pts, self.pending["R"], self.sel)

    def _redraw_keypoints(self):
        self._redraw_markers()
        # update rows in place: rebuilding the list mid-click would swallow double-clicks
        self.list.blockSignals(True)
        while self.list.count() > len(self.pairs):
            self.list.takeItem(self.list.count() - 1)
        for i, (a, b) in enumerate(self.pairs):
            res = f"   {self._res_um[i]:.2f} µm" if self.A is not None and len(self.pairs) > 3 else ""
            text = f"#{i + 1}  X({a[0]:.0f},{a[1]:.0f})  IF({b[0]:.0f},{b[1]:.0f}){res}"
            if i < self.list.count():
                self.list.item(i).setText(text)
            else:
                self.list.addItem(text)
        self.list.setCurrentRow(-1 if self.sel is None else self.sel)
        self.list.blockSignals(False)
        # re-map the drawn outlines right away; re-select which nuclei to draw a moment later
        if self.A is not None:
            self.R.outline.setTransform(qtransform(self.R.O @ self.A))
        self.out_timer.start()

    def _list_row_changed(self, row):
        self.sel = row if row >= 0 else None
        self._redraw_markers()              # leave the list alone so its double-click still fires

    def _nudge(self, dx, dy):
        """Arrow keys: move the locked ghost, else the selected keypoint's IF position (screen px)."""
        upp = float(np.mean(self.R.vb.viewPixelSize()))
        d = np.array([dx, dy]) * upp                                  # in IF scene units
        if self.a_ov.isChecked() and self.g_state == "locked":
            self.A_lock = np.linalg.inv(self.R.O) @ T(*d) @ self.R.O @ self.A_lock
            self._update_ghost()
        elif self.sel is not None:
            a, b = self.pairs[self.sel]
            self.pairs[self.sel] = (a, b + np.linalg.inv(self.R.O)[:2, :2] @ d)
            self._fit()
            self._redraw_keypoints()

    def _undo(self):
        if self.g_state == "locked":
            self._escape()
            return
        if self.pending["L"] is not None or self.pending["R"] is not None:
            self.pending = {"L": None, "R": None}
            self._redraw_keypoints()
        elif self.pairs:
            self.pairs.pop()
            self._pairs_changed()

    def _delete_selected(self):
        if self.sel is not None:
            self.pairs.pop(self.sel)
            self.sel = None
            self._pairs_changed()

    def _clear(self):
        self.pairs, self.pending, self.sel = [], {"L": None, "R": None}, None
        self._pairs_changed()

    def _jump_to_pair(self, item):
        a, b = self.pairs[self.list.row(item)]
        self._active = "L"                  # with linked views, only the left pane drives the sync
        self.L.center_on(a)
        self.R.center_on(b)                 # centre on the pair itself, not on the fit's estimate

    # ---------------------------------------------------------------- overlay placement (ghost)
    def _overlay_toggled(self, on):
        if on:
            self.a_kp.setChecked(False)
            self.g_state = "float"
            self._reload_ghost()
            self.statusBar().showMessage(
                "Overlay mode: move the mouse over the IF pane — the Xenium nuclei follow it. Click when they "
                "match ([ ] rotate, - = scale" + (", wheel zooms the IF)." if self.A is None else ")."))
        else:
            self.g_state = "off"
            self.statusBar().showMessage("Overlay mode off.")
        self._update_ghost()

    def _sync_ghost_state(self):
        if not self.a_ov.isChecked():
            self.g_state = "off"
        elif self.g_state == "float":
            self._reload_ghost()
        self._update_ghost()

    def _reload_ghost(self):
        """Ghost = Xenium nuclei currently visible in the left pane (re-read when it settles)."""
        if self.g_state != "float" or self.bounds is None:
            return
        if self._nuc_screen_px(self.L, self.L.O) < MIN_NUC_PX:
            res, n = None, -1
        else:
            res, n = self.bounds.select(*self.L.visible_raw_bbox())
        self.R.set_ghost(res)
        self._ghost_ok = res is not None
        if res is None:
            self.statusBar().showMessage(
                "No nuclei in the Xenium view — pan the Xenium pane onto tissue." if n == 0 else
                "Zoom the Xenium pane in to float its nuclei.")
        self._update_ghost()

    def _float_G(self):
        """Xenium px -> IF scene while floating: left-pane centre under the IF mouse position,
        scaled by the ratio of the two panes' screen zooms (plus the manual rotation/scale)."""
        if self.R.last_move is None:
            return None
        m = self.R.vb.mapSceneToView(self.R.last_move)
        c = self.L.vb.viewRect().center()
        k = float(np.mean(self.R.vb.viewPixelSize())) / float(np.mean(self.L.vb.viewPixelSize())) * self.g_scale
        return T(m.x(), m.y()) @ Rot(self.g_theta) @ S(k, k) @ T(-c.x(), -c.y()) @ self.L.O

    def _update_ghost(self):
        G = None
        if self.g_state == "float" and getattr(self, "_ghost_ok", False):
            G = self._float_G()
        elif self.g_state == "locked":
            G = self.R.O @ self.A_lock
        if G is None:
            self.R.ghost.hide()
        else:
            self.R.ghost.setTransform(qtransform(G))
            self.R.ghost.show()

    def _lock_ghost(self, x, y):
        if not getattr(self, "_ghost_ok", False) or self._float_G() is None:
            return
        self.A_lock = np.linalg.inv(self.R.O) @ self._float_G()
        self.lock_raw = np.array([x, y])
        self.g_state = "locked"
        self._update_ghost()
        self.statusBar().showMessage("Overlay locked (arrows nudge, [ ] rotate, - = scale, Esc unlocks). "
                                     "Now click the spot you are most confident about to place the keypoint.")

    def _ghost_adjust(self, rot=0.0, scale=1.0):
        if self.g_state == "off":
            return
        self.g_theta += rot
        self.g_scale *= scale
        if self.g_state == "locked":
            q = self.R.to_scene(self.lock_raw)[0]
            X = T(*q) @ Rot(rot) @ S(scale, scale) @ T(*-q)
            self.A_lock = np.linalg.inv(self.R.O) @ X @ self.R.O @ self.A_lock
        self._update_ghost()
        self.statusBar().showMessage(f"Ghost rotation {math.degrees(self.g_theta):+.1f}° · "
                                     f"extra scale {self.g_scale:.3f}")

    def _escape(self):
        if self.g_state == "locked":
            self.g_state = "float"
            self._reload_ghost()
            self.statusBar().showMessage("Overlay unlocked — it follows the mouse again.")
        elif self.pending["L"] is not None or self.pending["R"] is not None:
            self.pending = {"L": None, "R": None}
            self._redraw_keypoints()

    # ---------------------------------------------------------------- linking / cursor
    def _orient(self, F):
        self.R.set_orientation(F @ self.R.O)
        self._redraw_keypoints()
        self._update_ghost()

    def _link_toggled(self, on):
        if on:
            self._range_changed(self._active, force=True)

    def _range_changed(self, side, force=False):
        if self._syncing or self.A is None or not self.a_link.isChecked():
            return
        if side != self._active and not force:
            return
        src, dst = (self.L, self.R) if side == "L" else (self.R, self.L)
        M = self._scene_map(side)
        r = src.vb.viewRect()
        c = apply(M, [[r.center().x(), r.center().y()]])[0]
        s = math.sqrt(abs(np.linalg.det(M[:2, :2])))
        w, h = r.width() * s, r.height() * s
        self._syncing = True
        try:
            dst.vb.setRange(QtCore.QRectF(c[0] - w / 2, c[1] - h / 2, w, h), padding=0)
        finally:
            self._syncing = False

    def _mirror_toggled(self, on):
        if not on:
            self.L.set_cursor(None)
            self.R.set_cursor(None)

    def _hovered(self, side, x, y):
        self._active = side
        if side == "R" and self.g_state == "float":
            self._update_ghost()
        src, dst = (self.L, self.R) if side == "L" else (self.R, self.L)
        src.set_cursor(None)
        A = self._current_map()
        xen = np.array([x, y]) if side == "L" else (apply(np.linalg.inv(A), [[x, y]])[0]
                                                   if A is not None else None)
        if A is not None and self.a_cur.isChecked():
            other = apply(A, [[x, y]])[0] if side == "L" else xen
            dst.set_cursor(other)
        else:
            dst.set_cursor(None)
        msg = f"Xenium: ({xen[0]:.1f}, {xen[1]:.1f}) px = ({xen[0] * self.px_um:.1f}, {xen[1] * self.px_um:.1f}) µm" \
            if xen is not None else f"IF raw: ({x:.1f}, {y:.1f}) px"
        if self.a_kp.isChecked():
            msg = "[PAIR-CLICK MODE]  " + msg
        elif self.g_state in ("float", "locked"):
            msg = f"[OVERLAY {self.g_state.upper()} · keypoint {len(self.pairs) + 1}]  " + msg
        self.statusBar().showMessage(msg)

    # ---------------------------------------------------------------- nuclei overlay
    def _nuc_screen_px(self, pane, M):
        """Typical nucleus size in screen px on a pane, given M: Xenium px -> that pane's scene."""
        scale = math.sqrt(abs(np.linalg.det(M[:2, :2])))
        return self.bounds.typical_size * scale / float(np.mean(pane.vb.viewPixelSize()))

    def _update_outlines(self, *a, sides="LR"):
        if self.bounds is None or not self.a_nuc.isChecked():
            for side, pane in (("L", self.L), ("R", self.R)):
                pane.set_outline(None)
                self._out_box[side] = None
            return
        msgs = []
        for side in sides:
            pane = self.L if side == "L" else self.R
            if side == "R" and self.A is None:
                pane.set_outline(None)
                self._out_box[side] = None
                continue
            # vertices stay in Xenium px; the item transform M maps them onto the pane
            M = pane.O if side == "L" else pane.O @ self.A
            if self._nuc_screen_px(pane, M) < MIN_NUC_PX:
                pane.set_outline(None)
                self._out_box[side] = None
                msgs.append(f"{'left' if side == 'L' else 'right'}: zoom in to see nuclei")
                continue
            x0, y0, x1, y1 = bbox(apply(np.linalg.inv(M), rect_corners(pane.vb.viewRect())))
            box = self._out_box[side]
            if box and box[0] <= x0 and box[1] <= y0 and box[2] >= x1 and box[3] >= y1:
                pane.outline.setTransform(qtransform(M))      # still covered: just re-map
                continue
            px, py = (x1 - x0) * OUTLINE_PAD, (y1 - y0) * OUTLINE_PAD
            box = (x0 - px, y0 - py, x1 + px, y1 + py)
            res, n = self.bounds.select(*box)
            pane.set_outline(res, M)
            self._out_box[side] = box if res is not None else None
            if res is None and n:
                msgs.append(f"{'left' if side == 'L' else 'right'}: {n} nuclei in view — zoom in to draw")
        if msgs:
            self.statusBar().showMessage(" · ".join(msgs))

    def closeEvent(self, ev):
        super().closeEvent(ev)
        self.closed.emit()

    # ---------------------------------------------------------------- I/O
    def _save(self):
        if self.A is None:
            QtWidgets.QMessageBox.information(self, "Save", "Place at least 3 keypoint pairs first.")
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save alignment to folder")
        if d:
            self.save_to(d)
            self.statusBar().showMessage(f"Saved to {d}")

    def save_to(self, d):
        X = np.array([p[0] for p in self.pairs])
        U = np.array([p[1] for p in self.pairs])
        res = self._res_um if len(self.pairs) > 3 else np.zeros(len(X))
        with open(os.path.join(d, "keypoints.csv"), "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(KP_COLS)
            wr.writerows([repr(float(v)) for v in row] for row in np.column_stack([X, U]))
        np.savetxt(os.path.join(d, "xenium_px_to_if_px.csv"), self.A, delimiter=",", fmt="%.10g")
        np.savetxt(os.path.join(d, "if_px_to_xenium_px.csv"), np.linalg.inv(self.A), delimiter=",", fmt="%.10g")
        meta = {
            "xenium_image": os.path.abspath(self.xen.path),
            "if_image": os.path.abspath(self.ifim.path),
            "xenium_pixel_size_um": self.px_um,
            "convention": "3x3 matrices act on column vectors [x, y, 1]; x = column, y = row, "
                          "level-0 pixels of each image; IF coords are RAW (before display flips).",
            "xenium_px_to_if_px": self.A.tolist(),
            "if_px_to_xenium_px": np.linalg.inv(self.A).tolist(),
            "display_orientation_if": self.R.O.tolist(),
            "n_pairs": len(self.pairs),
            "keypoint_residuals_um": res.tolist(),
        }
        with open(os.path.join(d, "alignment.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load keypoints.csv", "", "CSV (*.csv)")
        if p:
            self.load_keypoints(p)

    def load_keypoints(self, p):
        """keypoints.csv in Xenium Explorer format (fixedX,fixedY,alignmentX,alignmentY);
        files saved by earlier versions of this tool are accepted too."""
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = [h.strip() for h in reader.fieldnames or []]
            reader.fieldnames = header
            cols = next((c for c in (KP_COLS, KP_COLS_OLD) if set(c) <= set(header)), None)
            if cols is None:
                QtWidgets.QMessageBox.warning(self, "Load keypoints",
                                              f"Expected columns {', '.join(KP_COLS)} — got {', '.join(header)}")
                return
            rows = [[float(r[c]) for c in cols] for r in reader if any((v or "").strip() for v in r.values())]
        self.pairs = [(np.array(r[:2]), np.array(r[2:])) for r in rows]
        self.pending, self.sel = {"L": None, "R": None}, None
        self._pairs_changed()


# ----------------------------------------------------------------------------- start window
class StartDialog(QtWidgets.QDialog):
    """Pick the input files, then "Start Aligner". Last-used paths are remembered."""

    IMAGES = "Images (*.ome.tif *.ome.tiff *.tif *.tiff);;All files (*)"
    FIELDS = [  # key, label, hint, file filter, required
        ("xenium", "Xenium morphology image",
         "e.g. morphology_focus/ch0000_dapi.ome.tif — the other channels in the folder are found automatically",
         IMAGES, True),
        ("if_path", "Post-Xenium image (IF / H&E)", "TIFF / OME-TIFF; pyramidal is much faster", IMAGES, True),
        ("boundaries", "Nucleus segmentation",
         "nucleus_boundaries.parquet (or cell_boundaries.parquet) — needed for the nuclei overlay",
         "Parquet (*.parquet);;All files (*)", False),
        ("keypoints", "Keypoints to resume (optional)",
         "keypoints.csv from this tool or Xenium Explorer (fixedX,fixedY,alignmentX,alignmentY)",
         "CSV (*.csv);;All files (*)", False),
    ]

    def __init__(self, initial=None):
        super().__init__()
        self.setWindowTitle("st_align — choose files")
        self.settings = QtCore.QSettings("st_align", "st_align")
        self.edits = {}
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, (key, label, hint, filt, required) in enumerate(self.FIELDS):
            lbl = QtWidgets.QLabel(f"<b>{label}</b><br>"
                                   f"<span style='color:gray'>{hint}</span>")
            lbl.setWordWrap(True)
            edit = QtWidgets.QLineEdit()
            edit.setMinimumWidth(480)
            edit.setPlaceholderText("required" if required else "optional")
            edit.textChanged.connect(self._validate)
            btn = QtWidgets.QPushButton("Browse…")
            btn.clicked.connect(lambda _=False, k=key, l=label, f=filt: self._browse(k, l, f))
            grid.addWidget(lbl, 2 * row, 0, 1, 3)
            grid.addWidget(edit, 2 * row + 1, 0, 1, 2)
            grid.addWidget(btn, 2 * row + 1, 2)
            self.edits[key] = edit
        self.edits["xenium"].textChanged.connect(self._guess_boundaries)

        self.msg = QtWidgets.QLabel("")
        self.msg.setStyleSheet("color: #c0392b")
        self.msg.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox()
        self.start = buttons.addButton("Start Aligner", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.start.setDefault(True)
        buttons.addButton("Quit", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self.msg)
        lay.addWidget(buttons)

        for key, edit in self.edits.items():                     # command line first, else last session
            edit.setText((initial or {}).get(key) or self.settings.value(key, "", str))
        self._guess_boundaries(self.edits["xenium"].text())     # after all fields are filled
        self._validate()

    def values(self):
        return {k: e.text().strip() or None for k, e in self.edits.items()}

    def remember(self):
        for k, v in self.values().items():
            self.settings.setValue(k, v or "")

    def _browse(self, key, label, filt):
        cur = self.edits[key].text().strip()
        start = cur or next((v for v in self.values().values() if v), "")
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, label, os.path.dirname(start) if start else "", filt)
        if p:
            self.edits[key].setText(p)

    def _guess_boundaries(self, path):
        """Xenium output layout: <out>/morphology_focus/chXXXX.ome.tif + <out>/nucleus_boundaries.parquet."""
        if self.edits["boundaries"].text().strip() or not path.strip():
            return
        d = os.path.dirname(path.strip())
        for cand in (os.path.join(os.path.dirname(d), "nucleus_boundaries.parquet"),
                     os.path.join(d, "nucleus_boundaries.parquet")):
            if os.path.isfile(cand):
                self.edits["boundaries"].setText(cand)
                return

    def _validate(self, *a):
        problems = []
        for key, label, _, _, required in self.FIELDS:
            v = self.edits[key].text().strip()
            if not v and required:
                problems.append(f"{label}: required")
            elif v and not os.path.isfile(v):
                problems.append(f"{label}: file not found")
        if not problems and not self.edits["boundaries"].text().strip():
            self.msg.setStyleSheet("color: gray")
            self.msg.setText("No segmentation file: overlay placement and the nuclei overlay will be unavailable.")
        else:
            self.msg.setStyleSheet("color: #c0392b")
            self.msg.setText(" · ".join(problems))
        self.start.setEnabled(not problems)


def launch(xenium, if_path, boundaries=None, keypoints=None, pixel_size=None):
    w = MainWindow(xenium, if_path, boundaries, pixel_size)
    if keypoints:
        w.load_keypoints(keypoints)
    w.show()
    return w


class Session(QtCore.QObject):
    """Start window -> aligner -> start window ... ; quitting the start window (or the app) quits."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.w = None
        self.quitting = False
        self.dlg = StartDialog({"xenium": args.xenium, "if_path": args.if_path,
                                "boundaries": args.boundaries, "keypoints": args.keypoints})
        self.dlg.accepted.connect(lambda: self._launch(self.dlg.values(), from_dlg=True))
        self.dlg.rejected.connect(QtWidgets.QApplication.quit)
        QtWidgets.QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, ev):                 # app-wide quit (Cmd+Q / "Quit" menu): don't reopen
        if ev.type() == QtCore.QEvent.Type.Quit and obj is QtWidgets.QApplication.instance():
            self.quitting = True
        return False

    def start(self):
        a = self.args
        if a.xenium and a.if_path:                  # everything given on the command line: start directly
            self._launch({"xenium": a.xenium, "if_path": a.if_path, "boundaries": a.boundaries,
                          "keypoints": a.keypoints}, from_dlg=False)
        else:
            self.dlg.show()

    def _launch(self, files, from_dlg):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.w = launch(**files, pixel_size=self.args.pixel_size)
        except Exception as e:                      # unreadable file etc.: back to the start window
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(None, "Could not start", f"{type(e).__name__}: {e}")
            self.dlg.show()
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        if from_dlg:
            self.dlg.remember()
        self.w.closed.connect(self._aligner_closed)

    def _aligner_closed(self):
        self.w.deleteLater()                        # free the images
        self.w = None
        # deferred, so an app-wide quit that is closing all windows finishes first
        QtCore.QTimer.singleShot(0, lambda: None if self.quitting else self.dlg.show())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xenium", help="Xenium morphology OME-TIFF (DAPI)")
    ap.add_argument("--if", dest="if_path", help="post-Xenium IF / H&E image (TIFF/OME-TIFF)")
    ap.add_argument("--boundaries", help="nucleus_boundaries.parquet (optional, for the overlay)")
    ap.add_argument("--pixel-size", type=float, help=f"Xenium µm/px (default: from OME, else {XENIUM_PX_UM})")
    ap.add_argument("--keypoints", help="keypoints.csv (Xenium Explorer format: fixedX,fixedY,alignmentX,alignmentY)")
    args = ap.parse_args()

    app = pg.mkQApp("st_align")
    app.setQuitOnLastWindowClosed(False)            # closing the aligner returns to the start window
    session = Session(args)
    session.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
