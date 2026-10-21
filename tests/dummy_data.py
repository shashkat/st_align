"""
Small synthetic Xenium-like dataset with a known alignment, for tests and for trying the app
without real data:

    python tests/dummy_data.py OUT_DIR
    python main.py --xenium OUT_DIR/morphology_focus/ch0000_dapi.ome.tif \\
                   --if OUT_DIR/post_xenium_if.ome.tif --boundaries OUT_DIR/nucleus_boundaries.parquet

Writes (laid out like a Xenium output folder):
    morphology_focus/ch0000_dapi.ome.tif  pyramidal OME-TIFF, CYX = (DAPI, membrane), uint16, JPEG 2000
    nucleus_boundaries.parquet            nucleus polygons in µm (cell_id, vertex_x, vertex_y, label_id)
    post_xenium_if.ome.tif                pyramidal RGB OME-TIFF: the DAPI image moved by TRUE_A
                                          (mirrored, rotated ~93°, scaled 0.5), re-noised
    true_alignment.json                   ground truth: TRUE_A (Xenium px -> raw IF px)

Deterministic (fixed seed). Generating takes a few seconds.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile

PX_UM = 0.2125                  # Xenium pixel size (µm)
XEN_SHAPE = (2400, 3000)        # (H, W) Xenium level-0 px
N_LEVELS = 3                    # pyramid levels written for both images
VERTS = 16                      # polygon vertices per nucleus
SEED = 7


def _true_affine():
    """Xenium px -> raw IF px: mirror, rotate 93°, scale 0.5, then shift into the IF frame."""
    th = math.radians(93)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    L = 0.5 * R @ np.array([[1, 0], [0, -1]])                    # mirrored, so a flip is needed
    H, W = XEN_SHAPE
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) @ L.T
    A = np.eye(3)
    A[:2, :2] = L
    A[:2, 2] = -corners.min(0) + 60                               # 60 px margin
    if_shape = tuple(int(v) for v in np.ceil(corners.max(0) - corners.min(0) + 120)[::-1])   # (H, W)
    return A, if_shape


TRUE_A, IF_SHAPE = _true_affine()


def make_nuclei(rng):
    """Nucleus centres inside an irregular tissue mask, plus elliptical polygons (Xenium px)."""
    H, W = XEN_SHAPE
    noise = cv2.GaussianBlur(rng.standard_normal((H // 8, W // 8)).astype(np.float32), (0, 0), 6)
    yy, xx = np.mgrid[0:H // 8, 0:W // 8]
    ell = ((xx - W / 16) / (W / 20)) ** 2 + ((yy - H / 16) / (H / 22)) ** 2
    mask = cv2.resize(((1 - ell) + 4 * noise > 0.15).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    # jittered grid -> no overlapping nuclei
    step = 46
    gy, gx = np.mgrid[step:H - step:step, step:W - step:step]
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(float) + rng.uniform(-12, 12, (gx.size, 2))
    keep = mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0
    keep &= rng.random(len(pts)) < 0.8                            # gaps, so the pattern is not periodic
    pts = pts[keep]
    polys = []
    t = np.linspace(0, 2 * np.pi, VERTS, endpoint=False)
    for cx, cy in pts:
        a, b, phi = rng.uniform(10, 17), rng.uniform(7, 12), rng.uniform(0, np.pi)
        x = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        y = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
        polys.append(np.column_stack([x, y]))
    return mask, pts, polys


def render_xenium(rng, mask, polys):
    """(2, H, W) uint16: DAPI (bright nuclei on faint tissue texture) and a membrane-like channel."""
    H, W = XEN_SHAPE
    dapi = np.zeros((H, W), np.float32)
    for p in polys:
        cv2.fillPoly(dapi, [np.round(p).astype(np.int32)], float(rng.uniform(600, 1400)))
    texture = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3) * 60 + 120
    dapi = cv2.GaussianBlur(dapi, (0, 0), 1.5) + texture * mask + rng.normal(30, 12, (H, W))
    membrane = cv2.GaussianBlur((cv2.Canny(mask * 255, 50, 150) > 0).astype(np.float32) * 800, (0, 0), 4)
    membrane += cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 2) * 40 * mask + 50
    return np.clip(np.stack([dapi, membrane]), 0, 65535).astype(np.uint16)


def render_if(rng, dapi):
    """RGB uint8 post-Xenium 'IF': DAPI moved by TRUE_A, different contrast and noise, signal in blue."""
    h, w = IF_SHAPE
    moved = cv2.warpAffine(dapi.astype(np.float32), TRUE_A[:2], (w, h), flags=cv2.INTER_AREA)
    v = np.clip(moved / 1200, 0, 1) ** 0.8 * 200 + rng.normal(15, 6, (h, w))
    v = np.clip(cv2.GaussianBlur(v.astype(np.float32), (0, 0), 0.8), 0, 255)
    return np.stack([v * 0.35, v * 0.55, v], axis=-1).astype(np.uint8)


def write_pyramid(path, data, axes, px_um, channel_names=None, compression="zlib"):
    """Pyramidal OME-TIFF (sub-IFDs), each level half the previous, like Xenium / bfconvert output."""
    meta = {"axes": axes, "PhysicalSizeX": px_um, "PhysicalSizeY": px_um}
    if channel_names:
        meta["Channel"] = {"Name": channel_names}
    yx = (axes.index("Y"), axes.index("X"))
    opts = dict(tile=(256, 256), compression=compression,
                photometric="rgb" if "S" in axes else "minisblack")
    with tifffile.TiffWriter(path, ome=True) as tif:
        tif.write(data, subifds=N_LEVELS - 1, metadata=meta, **opts)
        level = data
        for _ in range(N_LEVELS - 1):
            sl = [slice(None)] * level.ndim
            sl[yx[0]] = sl[yx[1]] = slice(None, None, 2)
            level = level[tuple(sl)]
            tif.write(level, subfiletype=1, **opts)


def generate(out_dir):
    """Write the dataset to out_dir; returns a dict of the written paths and the ground truth."""
    rng = np.random.default_rng(SEED)
    mask, centres, polys = make_nuclei(rng)
    xen = render_xenium(rng, mask, polys)
    rgb = render_if(rng, xen[0])

    os.makedirs(os.path.join(out_dir, "morphology_focus"), exist_ok=True)
    paths = {
        "xenium": os.path.join(out_dir, "morphology_focus", "ch0000_dapi.ome.tif"),
        "if_path": os.path.join(out_dir, "post_xenium_if.ome.tif"),
        "boundaries": os.path.join(out_dir, "nucleus_boundaries.parquet"),
        "truth": os.path.join(out_dir, "true_alignment.json"),
    }
    write_pyramid(paths["xenium"], xen, "CYX", PX_UM, ["DAPI", "Membrane"], compression="jpeg2000")
    write_pyramid(paths["if_path"], rgb, "YXS", PX_UM * 2)
    ids = np.repeat([f"cell{i:05d}" for i in range(len(polys))], VERTS)
    verts = np.concatenate(polys) * PX_UM                         # Xenium stores µm
    pd.DataFrame({"cell_id": ids, "vertex_x": verts[:, 0].astype(np.float32),
                  "vertex_y": verts[:, 1].astype(np.float32),
                  "label_id": np.repeat(np.arange(1, len(polys) + 1), VERTS)}).to_parquet(paths["boundaries"])
    with open(paths["truth"], "w") as f:
        json.dump({"xenium_px_to_if_px": TRUE_A.tolist(), "xenium_pixel_size_um": PX_UM,
                   "n_nuclei": len(polys)}, f, indent=2)
    return {**paths, "A": TRUE_A, "centres": centres, "n_nuclei": len(polys)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    info = generate(sys.argv[1])
    print(f"wrote {info['n_nuclei']} nuclei to {sys.argv[1]}")
