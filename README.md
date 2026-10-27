# st_align

A desktop tool for aligning **10x Genomics Xenium** spatial transcriptomics data with an image of the
same tissue section taken afterwards, such as **post-Xenium immunofluorescence (IF)** or **H&E**.

You place matching keypoints between the Xenium morphology (DAPI) image and the post-Xenium image,
either automatically or by hand. st_align fits an affine transform and lets you check it against the
Xenium nucleus segmentation drawn over the post-Xenium image. Keypoints are saved in
**Xenium Explorer's `keypoints.csv` format**, so you can move them between the two tools.

## Features

- **Two linked image panes:** the Xenium image (fixed) and the post-Xenium image (moving). Pyramidal
  OME-TIFFs are read lazily, so large whole-slide images stay responsive.
- **Automatic alignment:** SIFT feature matching, as in [InSituPy](https://pypi.org/project/insitupy-spatial/),
  followed by per-keypoint refinement with local cross-correlation.
- **Overlay placement:** the Xenium nuclei float over the IF/H&E image and follow your mouse. Line them
  up with the nuclei underneath, click to lock, then click to place a keypoint.
- **Live QC:** once 3 keypoints exist, the views are linked and the nucleus outlines are drawn through
  the fitted transform. Nudging a keypoint with the arrow keys updates the overlay instantly.
  Per-keypoint and RMS residuals are shown in µm.
- **Display controls** for each image: min/max contrast, gamma, channel selection, and flip/rotate for
  the post-Xenium image (display only; the exported coordinates are always in the raw image).
- **Compatible with Xenium Explorer:** keypoints exported from Xenium Explorer's image alignment can be
  loaded and refined here, and keypoints saved here can be imported there.

## Installation

Requires Python ≥ 3.12.

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/shashkat/st_align.git
cd st_align
uv sync
```

Or with pip, in an environment of your choice:

```bash
pip install pyqt6 pyqtgraph tifffile zarr imagecodecs pandas pyarrow numpy opencv-python-headless
```

The tested minimum versions are listed in [`pyproject.toml`](pyproject.toml). Use
`opencv-python-headless` rather than `opencv-python`: the latter bundles its own Qt, which can clash
with PyQt6.

## Running

```bash
uv run main.py          # or: python main.py
```

This opens a start window where you choose:

| Input | Required | Example |
|---|---|---|
| Xenium morphology image | yes | `output-XETG…/morphology_focus/ch0000_dapi.ome.tif` |
| Post-Xenium image (IF / H&E) | yes | any TIFF / OME-TIFF; pyramidal is much faster |
| Nucleus segmentation | no, but needed for the nuclei overlay and overlay placement | `output-XETG…/nucleus_boundaries.parquet` |
| Keypoints to resume | no | a `keypoints.csv` from st_align or Xenium Explorer |

If you pick an image inside a Xenium `morphology_focus/` folder, `nucleus_boundaries.parquet` is filled
in automatically from the output folder. The start window remembers the files you used last.

Closing the aligner window takes you back to the start window. Quitting the start window quits the app.

### Command line

You can skip the start window by passing the files directly:

```bash
uv run main.py --xenium output-XETG.../morphology_focus/ch0000_dapi.ome.tif \
               --if post_xenium_IF.ome.tif \
               --boundaries output-XETG.../nucleus_boundaries.parquet \
               --keypoints keypoints.csv          # optional
```

| Option | Meaning |
|---|---|
| `--xenium` | Xenium morphology OME-TIFF |
| `--if` | post-Xenium IF / H&E image |
| `--boundaries` | `nucleus_boundaries.parquet` (or `cell_boundaries.parquet`) |
| `--keypoints` | keypoints to load (Xenium Explorer format) |
| `--pixel-size` | Xenium µm per pixel (default: read from the OME metadata, else 0.2125) |

If you give only some of the files, the start window opens with those filled in.

> **About the Xenium `morphology_focus/` folder:** each `chXXXX_*.ome.tif` file holds one channel, but
> its OME metadata lists the others. Opening any one of them loads all channels (DAPI, boundary stain,
> …), which you can switch between in the Xenium pane. Keep the files together under their original
> names.

## Workflow

1. **Orient the post-Xenium image.** Use *IF: flip ↔ / flip ↕ / rotate 90°* until it looks like the
   Xenium image. This only changes the display.
2. **Place keypoints**, in any combination of these ways:
   - **Auto-align (A).** Pick the IF channel to match on (for H&E, the hematoxylin channel) and how many
     keypoints to place. Only the mirror state of your orientation matters, since rotation is found
     automatically; if the image only matches mirrored, the IF pane is flipped for you.
   - **Overlay placement (O).** Zoom both panes to a similar scale. The nuclei visible in the Xenium pane
     float over the IF image, centred on your cursor. Line them up with the nuclei underneath, using
     `[` / `]` to rotate and `-` / `=` to scale, then click to lock. Arrow keys nudge the locked overlay;
     `Esc` unlocks it. Click the spot you're most confident about to place the keypoint. Pan the Xenium
     pane to a different region and repeat.
   - **Pair clicks (K).** Click a point in the Xenium pane and the matching point in the IF pane.
3. **Review.** After 3 keypoints an affine transform is fitted, the panes zoom and pan together, and the
   nucleus outlines (red) are drawn through the fit. Zoom in across the tissue and check that they sit
   on the nuclei.
   - Select a keypoint in the list (double-click to jump to it) and nudge it with the arrow keys
     (`Shift` for 10×). The overlay updates as you go.
   - Delete bad keypoints with `Del`, and add more where the fit is off.
   - The keypoint list shows each keypoint's residual. The panel below it shows the RMS residual, plus
     the fitted scale and rotation.
4. **Save (Ctrl+S)** to a folder (see [Output](#output)).

### Keyboard shortcuts

| Key | Action |
|---|---|
| `A` | Auto-align… |
| `O` | Overlay placement mode |
| `K` | Pair-click mode |
| `L` | Link / unlink the panes (available once fitted) |
| `N` | Show / hide nucleus outlines |
| `M` | Show / hide the mirrored cursor (yellow cross in the other pane) |
| Arrow keys | Nudge the locked overlay, or the selected keypoint (`Shift` = 10×) |
| `[` / `]` | Rotate the floating overlay by 0.5° |
| `-` / `=` | Scale the floating overlay by 0.5% |
| `Esc` | Unlock the overlay |
| `Ctrl+Z` | Undo the last keypoint |
| `Del` | Delete the selected keypoint |
| `Ctrl+S` | Save |

Each pane also has a contrast bar: drag the edges of the histogram to set min/max (double-click it for
the full range), use the γ slider for gamma (double-click it to reset), or press *Auto*. Nucleus
outlines are only drawn once you've zoomed in far enough to see them.

## Output

*Save* writes four files to the chosen folder (it needs at least 3 keypoints):

| File | Contents |
|---|---|
| `keypoints.csv` | `fixedX,fixedY,alignmentX,alignmentY`, the Xenium Explorer format. Can be imported into Xenium Explorer or loaded back into st_align. |
| `xenium_px_to_if_px.csv` | 3×3 affine matrix: Xenium pixels → post-Xenium image pixels |
| `if_px_to_xenium_px.csv` | its inverse |
| `alignment.json` | both matrices, image paths, Xenium pixel size, display orientation, per-keypoint residuals (µm) |

**Coordinate conventions:**
- `fixedX/fixedY` are **level-0 pixels of the Xenium morphology image**: x = column, y = row. Divide
  Xenium µm coordinates by the pixel size (0.2125 µm) to get them.
- `alignmentX/alignmentY` are **level-0 pixels of the raw post-Xenium image**, before any display
  flip or rotation.
- The matrices act on column vectors `[x, y, 1]`.

For example, to map Xenium cell centroids into the post-Xenium image:

```python
import json
import numpy as np
import pandas as pd

aln = json.load(open("alignment/alignment.json"))
A = np.array(aln["xenium_px_to_if_px"])              # 3x3, Xenium px -> IF px
px = aln["xenium_pixel_size_um"]

cells = pd.read_parquet("output-XETG.../cells.parquet")
xy = cells[["x_centroid", "y_centroid"]].to_numpy() / px          # µm -> Xenium px
if_xy = xy @ A[:2, :2].T + A[:2, 2]                               # -> IF image px (x, y)
```

To resample the post-Xenium image onto the Xenium pixel grid (at full resolution this needs a lot of
memory; downsample first for large slides):

```python
import cv2
warped = cv2.warpAffine(if_image, A[:2], (xenium_width, xenium_height),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
```

## Automatic alignment: details and limitations

1. **Coarse match:** both images are resized to 4000 px and contrast-clipped. SIFT features are matched
   with FLANN and Lowe's ratio test, and a rotation + uniform scale + translation transform is fitted
   with RANSAC.
2. **Refinement:** well-spread inlier matches are re-located by normalised cross-correlation of a
   ±80 µm Xenium patch against the post-Xenium image at ~0.4 µm/pixel. Points that don't correlate
   well are dropped.

On a test section, keypoints from st_align's Auto-align gave a transform within a median of
**0.48 µm** (max 1.2 µm) of one fitted to keypoints placed by hand in Xenium Explorer. InSituPy's
registration, given the same correctly oriented input, came within 1.0 µm (max 1.9 µm).

Feature matching needs enough shared, clean structure in both images. On very noisy post-Xenium images
it can fail. In that case st_align reports the failure and places nothing; use overlay placement
instead. Always review the result before saving.

## Tips

- **Use pyramidal (tiled, multi-resolution) TIFFs** for the post-Xenium image. Flat TIFFs work, but
  every zoom level has to be computed from full resolution.
- **Use a recent `tifffile`.** Older versions can't read the per-file pyramids of Xenium's multi-file
  `morphology_focus/` images and fall back to full resolution, which is very slow.
- **Use DAPI as the Xenium channel** for auto-alignment, and the nuclear channel of the IF image
  (or hematoxylin for H&E).
- **The transform is affine** (rotation, scale, shear, translation). Local tissue deformation between
  Xenium and post-Xenium imaging shows up as larger residuals in some regions. Check the residuals and
  the overlay across the whole section.

## Development and testing

The test suite uses [pytest](https://pytest.org) and [pytest-qt](https://pytest-qt.readthedocs.io). GUI
tests run headless (`QT_QPA_PLATFORM=offscreen`), so no display is needed.

```bash
uv sync                 # installs the dev dependencies (pytest, pytest-qt) too
uv run pytest           # whole suite, ~20 s
uv run pytest -m "not gui"                    # only the non-GUI tests
uv run pytest tests/test_gui_overlay.py -k fit   # one file / matching tests
```

**Test data is generated, not stored.** [`tests/dummy_data.py`](tests/dummy_data.py) builds a small,
deterministic Xenium-like dataset with a **known ground-truth alignment**: a pyramidal 2-channel DAPI
OME-TIFF in a `morphology_focus/` layout, `nucleus_boundaries.parquet`, and an RGB "post-Xenium IF"
image made by moving the DAPI image through a mirrored, rotated and scaled affine. Tests can therefore
check alignment accuracy against the truth. The dataset is generated once per test run (a few seconds).
You can also write it to disk to try the app without real data:

```bash
uv run tests/dummy_data.py /tmp/st_align_dummy
uv run main.py --xenium /tmp/st_align_dummy/morphology_focus/ch0000_dapi.ome.tif \
               --if /tmp/st_align_dummy/post_xenium_if.ome.tif \
               --boundaries /tmp/st_align_dummy/nucleus_boundaries.parquet
```

**What's covered:**

| File | Covers |
|---|---|
| `test_geometry.py` | matrix conventions, Qt transform conversion, orientation matrices |
| `test_images.py` | pyramidal / multi-channel / RGB OME-TIFF reading |
| `test_boundaries.py` | parquet loading, spatial selection vs. brute force, typical nucleus size |
| `test_autoalign.py` | automatic alignment against the true transform, mirror retry, clean failure on noise, helpers |
| `test_gui_keypoints.py` | fitting and residuals, save/load (Xenium Explorer format, byte-exact round trip), nudging, selection, list double-click |
| `test_gui_overlay.py` | overlay placement end to end (recovers the true transform), ghost scale and adjustments, behaviour after the fit |
| `test_gui_display.py` | contrast and gamma, channel switching, when nucleus outlines are drawn and reused, mirrored cursor |
| `test_gui_start.py` | start window validation, auto-fill and remembered files; start window ↔ aligner cycle and quitting |
| `test_gui_autoalign.py` | Auto-align from the window: options dialog, mirroring, replace / keep, failures |

**Conventions for changes** (for people and coding agents):
- Run `uv run pytest` before and after a change, and add tests for new behaviour. When fixing a bug,
  add a test that fails without the fix.
- GUI tests drive the app with real Qt mouse and key events through the helpers in
  [`tests/helpers.py`](tests/helpers.py) (`click`, `move`, `key`, `set_view`, `double_click_list_row`, …).
- Tests must not touch the user's real settings: `conftest.py` redirects `main.app_settings()` to a
  temporary file, and records message boxes in the `messages` fixture instead of showing them.
- Coordinates are level-0 **Xenium px** and **raw IF px**. Display orientation (`Pane.O`) is display
  only. 3×3 matrices act on column vectors `[x, y, 1]`.

## License

[MIT](LICENSE)
