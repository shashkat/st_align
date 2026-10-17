#!/usr/bin/env python3
"""
xalign — two-pane prototype for aligning a post-Xenium IF image to the Xenium DAPI image.

Left pane  : Xenium morphology image (fixed), e.g. morphology_focus/morphology_focus_0000.ome.tif
Right pane : post-Xenium IF image (moving), any OME-TIFF / TIFF (pyramidal strongly recommended)

Workflow
  1. Flip / rotate the IF pane until it looks like the DAPI pane (display only; the raw IF
     pixel coordinates are what get exported, so you never have to flip the image in code).
  2. Press K (or the toolbar button) and click matching points: one click in each pane = one pair.
     Order doesn't matter. Plain drags still pan; wheel zooms.
  3. After 3 pairs an affine is fitted automatically. Press L to link the panes: moving either one
     moves the other to the corresponding region, and a yellow cross shows where your cursor maps.
  4. Press N to overlay Xenium nucleus outlines (needs --boundaries). On the IF pane they are drawn
     through the current affine, so this is the "does it fit" check.
  5. Save (Ctrl+S) writes keypoints + both affine directions to a folder of your choice.

Keys:  K keypoint mode · L link · N nuclei · Ctrl+Z undo · Del delete selected pair · Ctrl+S save
       Double-click a pair in the list to jump to it.

Run:
  python xalign.py --xenium morphology_focus_0000.ome.tif --if post_if.ome.tif \
                   --boundaries nucleus_boundaries.parquet
"""
import argparse
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
MAX_CELLS = 25000        # max nucleus outlines drawn at once
XENIUM_PX_UM = 0.2125    # Xenium morphology level-0 pixel size (µm)


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
        self.codes = codes
        self.x = df["vertex_x"].to_numpy(float) / px_um
        self.y = df["vertex_y"].to_numpy(float) / px_um
        n = codes.max() + 1
        cnt = np.bincount(codes, minlength=n)
        self.cx = np.bincount(codes, weights=self.x, minlength=n) / cnt
        self.cy = np.bincount(codes, weights=self.y, minlength=n) / cnt

    def select(self, x0, y0, x1, y1):
        sel = (self.cx >= x0) & (self.cx <= x1) & (self.cy >= y0) & (self.cy <= y1)
        n = int(sel.sum())
        if n == 0 or n > MAX_CELLS:
            return None, n
        vm = sel[self.codes]
        x, y, c = self.x[vm], self.y[vm], self.codes[vm]
        connect = np.zeros(len(c), dtype=bool)
        connect[:-1] = c[:-1] == c[1:]
        return (np.column_stack([x, y]), connect), n


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
        self.kp = pg.ScatterPlotItem(size=12, symbol="o", pen=pg.mkPen((0, 255, 120), width=2),
                                     brush=None)
        self.kp.setZValue(10)
        self.pending = pg.ScatterPlotItem(size=14, symbol="x", pen=pg.mkPen((255, 255, 0), width=2))
        self.pending.setZValue(11)
        self.cursor = pg.ScatterPlotItem(size=22, symbol="+", pen=pg.mkPen((255, 230, 0), width=2))
        self.cursor.setZValue(12)
        for it in (self.ov_item, self.tile_item, self.outline, self.kp, self.pending, self.cursor):
            self.vb.addItem(it)
        self.labels = []

        self.hist = None
        if not img.rgb:
            self.hist = pg.HistogramLUTWidget()
            self.hist.setMaximumWidth(110)

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

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.glw, 1)
        if self.hist:
            body.addWidget(self.hist)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(head)
        lay.addLayout(body, 1)
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
        self.ov_item.setImage(data, autoLevels=False)
        self.ov_item.setTransform(qtransform(self.O @ S(dx, dy)))
        self._ov_ds = max(dx, dy)
        if self.img.rgb:
            lv = (0, 255) if data.dtype == np.uint8 else tuple(np.percentile(data, [0.5, 99.8]))
            self.ov_item.setLevels(lv)
            self._rgb_levels = lv
        else:
            self.hist.setImageItem(self.ov_item)
            if auto_contrast:
                nz = data[data > 0]
                lo, hi = np.percentile(nz if nz.size else data, [1, 99.7])
                self.hist.item.setLevels(float(lo), float(max(hi, lo + 1)))
            if not getattr(self, "_hist_connected", False):
                self.hist.item.sigLevelsChanged.connect(self._sync_contrast)
                self.hist.item.sigLookupTableChanged.connect(self._sync_contrast)
                self._hist_connected = True
        self._tile_key = None

    def _sync_contrast(self, *a):
        if self.hist:
            self.tile_item.setLevels(self.hist.item.getLevels())
            self.tile_item.setLookupTable(self.hist.item.getLookupTable)

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
        self.tile_item.setImage(data, autoLevels=False)
        if img.rgb:
            self.tile_item.setLevels(self._rgb_levels)
        else:
            self._sync_contrast()
        self.tile_item.setTransform(qtransform(self.O @ T(c0 * dx, r0 * dy) @ S(dx, dy)))
        self.tile_item.show()
        self._tile_key = (lvl, c0, c1, r0, r1, img.channel)
        self.info.setText(f"level {lvl} (1/{max(dx, dy):.0f})")

    def _settle(self):
        self.refresh_tile()
        self.settled.emit()

    # -- overlays
    def set_keypoints(self, raw_pts, pending_raw=None):
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
        self.pending.setData(pos=self.to_scene(pending_raw) if pending_raw is not None else np.empty((0, 2)))

    def set_outline(self, scene_pts, connect):
        if scene_pts is None:
            self.outline.setData(x=np.array([]), y=np.array([]))
        else:
            self.outline.setData(x=scene_pts[:, 0], y=scene_pts[:, 1], connect=connect)

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
        p = self.vb.mapSceneToView(ev.scenePos())
        raw = self.to_raw([p.x(), p.y()])[0]
        self.clicked.emit(float(raw[0]), float(raw[1]))

    def _on_move(self, pos):
        if not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        raw = self.to_raw([p.x(), p.y()])[0]
        self.hovered.emit(float(raw[0]), float(raw[1]))


# ----------------------------------------------------------------------------- main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, xen_path, if_path, boundaries_path=None, px_um=None):
        super().__init__()
        self.setWindowTitle("xalign — Xenium ↔ post-Xenium IF")
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
        self.list.itemDoubleClicked.connect(self._jump_to_pair)
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
        self._syncing = False
        self._active = "L"

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
        self.a_kp = act("Add keypoints", self._mode_changed, "K", True)
        act("Undo", self._undo, "Ctrl+Z")
        act("Delete selected", self._delete_selected, "Delete")
        act("Clear", self._clear)
        tb.addSeparator()
        self.a_link = act("Link views", self._link_toggled, "L", True)
        self.a_link.setEnabled(False)
        self.a_nuc = act("Show nuclei", self._update_outlines, "N", True)
        self.a_nuc.setEnabled(self.bounds is not None)
        tb.addSeparator()
        act("Load keypoints…", self._load)
        act("Save…", self._save, "Ctrl+S")

        for side_, pane in (("L", self.L), ("R", self.R)):
            pane.clicked.connect(lambda x, y, s=side_: self._clicked(s, x, y))
            pane.hovered.connect(lambda x, y, s=side_: self._hovered(s, x, y))
            pane.settled.connect(self._update_outlines)
            pane.vb.sigRangeChanged.connect(lambda *a, s=side_: self._range_changed(s))

        self.resize(1700, 950)
        self.statusBar().showMessage("Orient the IF image, then press K and click matching points (left & right).")

    # ---------------------------------------------------------------- transforms
    def _scene_map(self, src):
        """3x3 mapping src-pane scene coords -> other-pane scene coords (needs self.A)."""
        M = self.R.O @ self.A @ np.linalg.inv(self.L.O)     # L scene -> R scene
        return M if src == "L" else np.linalg.inv(M)

    def _fit(self):
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

    # ---------------------------------------------------------------- keypoints
    def _mode_changed(self, on):
        self.statusBar().showMessage("Keypoint mode: click a point in each pane (any order)." if on
                                     else "Keypoint mode off.")

    def _clicked(self, side, x, y):
        if not self.a_kp.isChecked():
            return
        self.pending[side] = np.array([x, y])
        if self.pending["L"] is not None and self.pending["R"] is not None:
            self.pairs.append((self.pending["L"], self.pending["R"]))
            self.pending = {"L": None, "R": None}
            self._fit()
        self._redraw_keypoints()

    def _redraw_keypoints(self):
        self.L.set_keypoints(np.array([p[0] for p in self.pairs]).reshape(-1, 2), self.pending["L"])
        self.R.set_keypoints(np.array([p[1] for p in self.pairs]).reshape(-1, 2), self.pending["R"])
        self.list.clear()
        for i, (a, b) in enumerate(self.pairs):
            res = f"   {self._res_um[i]:.2f} µm" if self.A is not None and len(self.pairs) > 3 else ""
            self.list.addItem(f"#{i + 1}  X({a[0]:.0f},{a[1]:.0f})  IF({b[0]:.0f},{b[1]:.0f}){res}")
        self._update_outlines()

    def _undo(self):
        if self.pending["L"] is not None or self.pending["R"] is not None:
            self.pending = {"L": None, "R": None}
        elif self.pairs:
            self.pairs.pop()
            self._fit()
        self._redraw_keypoints()

    def _delete_selected(self):
        rows = sorted({self.list.row(i) for i in self.list.selectedItems()}, reverse=True)
        for r in rows:
            self.pairs.pop(r)
        if rows:
            self._fit()
            self._redraw_keypoints()

    def _clear(self):
        self.pairs, self.pending = [], {"L": None, "R": None}
        self._fit()
        self._redraw_keypoints()

    def _jump_to_pair(self, item):
        a, b = self.pairs[self.list.row(item)]
        self._active = "L"
        self.L.center_on(a)
        if not self.a_link.isChecked():
            self.R.center_on(b)

    # ---------------------------------------------------------------- linking / cursor
    def _orient(self, F):
        self.R.set_orientation(F @ self.R.O)
        self._redraw_keypoints()

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

    def _hovered(self, side, x, y):
        self._active = side
        src, dst = (self.L, self.R) if side == "L" else (self.R, self.L)
        src.set_cursor(None)
        xen = np.array([x, y]) if side == "L" else (apply(np.linalg.inv(self.A), [[x, y]])[0]
                                                   if self.A is not None else None)
        if self.A is not None:
            other = apply(self.A, [[x, y]])[0] if side == "L" else xen
            dst.set_cursor(other)
        else:
            dst.set_cursor(None)
        msg = f"Xenium: ({xen[0]:.1f}, {xen[1]:.1f}) px = ({xen[0] * self.px_um:.1f}, {xen[1] * self.px_um:.1f}) µm" \
            if xen is not None else f"IF raw: ({x:.1f}, {y:.1f}) px"
        if self.a_kp.isChecked():
            msg = "[KEYPOINT MODE]  " + msg
        self.statusBar().showMessage(msg)

    # ---------------------------------------------------------------- nuclei overlay
    def _update_outlines(self, *a):
        if self.bounds is None or not self.a_nuc.isChecked():
            self.L.set_outline(None, None)
            self.R.set_outline(None, None)
            return
        # left: directly in Xenium px
        res, n = self.bounds.select(*self.L.visible_raw_bbox())
        self.L.set_outline(None if res is None else self.L.to_scene(res[0]), None if res is None else res[1])
        msgs = [] if res is not None or n == 0 else [f"left: {n} nuclei in view — zoom in to draw"]
        # right: through the affine
        if self.A is None:
            self.R.set_outline(None, None)
        else:
            Ainv = np.linalg.inv(self.A)
            x0, y0, x1, y1 = self.R.visible_raw_bbox()
            xb = bbox(apply(Ainv, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
            res, n = self.bounds.select(*xb)
            if res is None:
                self.R.set_outline(None, None)
                if n:
                    msgs.append(f"right: {n} nuclei in view — zoom in to draw")
            else:
                self.R.set_outline(self.R.to_scene(apply(self.A, res[0])), res[1])
        if msgs:
            self.statusBar().showMessage(" · ".join(msgs))

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
        hdr = "xenium_x_px,xenium_y_px,if_x_px,if_y_px,xenium_x_um,xenium_y_um,residual_um"
        res = self._res_um if len(self.pairs) > 3 else np.zeros(len(X))
        np.savetxt(os.path.join(d, "keypoints.csv"),
                   np.column_stack([X, U, X * self.px_um, res]), delimiter=",", header=hdr,
                   comments="", fmt="%.4f")
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
        }
        with open(os.path.join(d, "alignment.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load keypoints.csv", "", "CSV (*.csv)")
        if p:
            self.load_keypoints(p)

    def load_keypoints(self, p):
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        self.pairs = [(r[0:2].copy(), r[2:4].copy()) for r in a]
        self.pending = {"L": None, "R": None}
        self._fit()
        self._redraw_keypoints()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xenium", required=True, help="Xenium morphology OME-TIFF (DAPI)")
    ap.add_argument("--if", dest="if_path", required=True, help="post-Xenium IF image (TIFF/OME-TIFF)")
    ap.add_argument("--boundaries", help="nucleus_boundaries.parquet (optional, for the overlay)")
    ap.add_argument("--pixel-size", type=float, help=f"Xenium µm/px (default: from OME, else {XENIUM_PX_UM})")
    ap.add_argument("--keypoints", help="keypoints.csv from a previous session")
    args = ap.parse_args()

    app = pg.mkQApp("xalign")
    w = MainWindow(args.xenium, args.if_path, args.boundaries, args.pixel_size)
    if args.keypoints:
        w.load_keypoints(args.keypoints)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
