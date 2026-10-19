"""
Automatic keypoint placement for st_align.

Two stages:
  1. Coarse (as in InSituPy): both images downscaled to <= COARSE_MAX_SIDE px, contrast clipped
     to the 20th-99th percentile, SIFT features matched with FLANN + Lowe's ratio test, and a
     similarity transform (rotation + uniform scale + translation) fitted with RANSAC.
  2. Refinement (not in InSituPy): well-spread RANSAC inliers are re-matched at ~REFINE_DS Xenium
     px per pixel by normalised cross-correlation of a Xenium patch against the IF warped through
     the coarse transform. Keypoints whose patches don't correlate well are dropped.

The IF image is matched in the orientation the user set in the IF pane (flips / 90° rotations),
so SIFT (which is not mirror-invariant) sees the two images the same way round. All returned
coordinates are level-0 Xenium px and RAW level-0 IF px, like the rest of st_align.

Needs OpenCV (pip install opencv-python-headless).
"""
import math

import numpy as np

COARSE_MAX_SIDE = 4000   # max side of the images used for feature matching (InSituPy's max_width)
RATIO = 0.7              # Lowe's ratio test
RANSAC_PX = 3.0          # RANSAC reprojection threshold, in coarse IF pixels
MIN_INLIERS = 10
REFINE_DS = 2            # refinement resolution, Xenium level-0 px per pixel (~0.42 µm)
PATCH = 192              # refinement template half-size (refinement px, ~±80 µm)
SEARCH = 48              # refinement search radius (refinement px, ~±20 µm)
MIN_SCORE = 0.3          # minimum correlation for a refined keypoint to be kept

IF_MODES = {             # how an RGB IF / H&E image is turned into one channel for matching
    "gray": "Grayscale (mean of R, G, B)",
    "r": "Red channel",
    "g": "Green channel",
    "b": "Blue channel",
    "hematoxylin": "Hematoxylin (H&E nuclei, colour deconvolution)",
}


class AutoAlignError(Exception):
    pass


def _cv2():
    try:
        import cv2
    except ImportError:
        raise AutoAlignError("Automatic alignment needs OpenCV:\n    pip install opencv-python-headless")
    return cv2


# ----------------------------------------------------------------------------- image helpers
def to_gray(data, mode="gray"):
    """One float32 channel for matching. Single-channel data passes through."""
    if data.ndim == 2:
        return data.astype(np.float32)
    rgb = data[..., :3].astype(np.float32)
    if mode in ("r", "g", "b"):
        return rgb[..., "rgb".index(mode)]
    if mode == "hematoxylin":
        # Ruifrok & Johnston colour deconvolution: optical density -> haematoxylin concentration
        maxv = 255.0 if data.dtype == np.uint8 else max(float(rgb.max()), 1.0)
        od = -np.log(np.clip(rgb, 1, None) / maxv)
        rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        rgb_from_hed /= np.linalg.norm(rgb_from_hed, axis=1, keepdims=True)
        h = od @ np.linalg.inv(rgb_from_hed)[:, 0]
        return np.clip(h, 0, None).astype(np.float32)
    return rgb.mean(-1)


def clip_u8(a, lo=20, hi=99):
    """Contrast clip to percentiles -> uint8 (InSituPy's 'clip' contrast adjustment)."""
    p_lo, p_hi = np.percentile(a, [lo, hi])
    return (np.clip((a - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def read_downsampled(img, max_side):
    """Whole image resized to max_side px (area-averaged; never upscaled).
    Returns (data, (dx, dy) level-0 px per pixel)."""
    cv2 = _cv2()
    # coarsest pyramid level still at least max_side px, then resize down to exactly max_side
    lvl = max((i for i, s in enumerate(img.shapes) if max(s) >= max_side), default=0)
    h, w = img.shapes[lvl]
    step = max(1, max(h, w) // (2 * max_side))       # cheap pre-decimation for huge single-level images
    data = img.read(lvl, 0, h, 0, w, step)
    dx, dy = img.ds[lvl][0] * step, img.ds[lvl][1] * step
    f = max_side / max(data.shape[:2])
    if f < 1:
        nh, nw = max(1, round(data.shape[0] * f)), max(1, round(data.shape[1] * f))
        dx, dy = dx * data.shape[1] / nw, dy * data.shape[0] / nh
        data = cv2.resize(data.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    return data, (dx, dy)


def read_region(img, x0, y0, x1, y1, target_ds):
    """Level-0 box at about target_ds level-0 px per pixel.
    Returns (data, (ox, oy) level-0 origin of pixel [0, 0], (dx, dy)), or None if outside the image."""
    lvl = max((i for i, (dx, dy) in enumerate(img.ds) if max(dx, dy) <= target_ds), default=0)
    step = max(1, int(round(target_ds / max(img.ds[lvl]))))
    dx, dy = img.ds[lvl]
    h, w = img.shapes[lvl]
    c0, c1 = max(0, int(x0 / dx)), min(w, int(math.ceil(x1 / dx)))
    r0, r1 = max(0, int(y0 / dy)), min(h, int(math.ceil(y1 / dy)))
    if c1 - c0 < 2 * step or r1 - r0 < 2 * step:
        return None                                   # box (almost) entirely outside the image
    return img.read(lvl, r0, r1, c0, c1, step), (c0 * dx, r0 * dy), (dx * step, dy * step)


def orient(a, O):
    """Apply a flip / 90°-rotation orientation (3x3, no translation) to an image array.
    Returns (oriented image, 3x3 mapping input pixel coords -> oriented pixel coords)."""
    cv2 = _cv2()
    h, w = a.shape[:2]
    L = O[:2, :2]
    c = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float) @ L.T
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = -c.min(0)
    size = tuple(int(v) for v in np.round(c.max(0) - c.min(0) + 1))
    return cv2.warpAffine(a, M[:2], size, flags=cv2.INTER_NEAREST), M


def _apply(M, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ M[:2, :2].T + M[:2, 2]


def _affine_lstsq(X, U):
    P, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), U, rcond=None)
    A = np.eye(3)
    A[0, :], A[1, :] = P[:, 0], P[:, 1]
    return A


def farthest_point_order(pts):
    """Indices of pts ordered so each next point is as far as possible from those before."""
    n = len(pts)
    if n == 0:
        return np.array([], int)
    order = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[order[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        if d[i] <= 0:
            break
        order.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(order)


# ----------------------------------------------------------------------------- stage 1: coarse
def coarse_match(xen, ifim, if_orient, if_mode="gray", progress=None):
    """SIFT + FLANN + RANSAC similarity on downscaled images.
    Returns dict with A (Xenium px -> IF raw px), inlier point arrays (level-0) and counts."""
    cv2 = _cv2()
    say = progress or (lambda *a: None)

    say("Reading images…", 0.05)
    xa, (xdx, xdy) = read_downsampled(xen, COARSE_MAX_SIDE)
    ua, (udx, udy) = read_downsampled(ifim, COARSE_MAX_SIDE)
    xg = clip_u8(to_gray(xa))
    ug, M_or = orient(clip_u8(to_gray(ua, if_mode)), if_orient)

    say("Detecting features (SIFT)…", 0.15)
    sift = cv2.SIFT_create()
    kx, dxs = sift.detectAndCompute(xg, None)
    ku, dus = sift.detectAndCompute(ug, None)
    if dxs is None or dus is None or len(kx) < 3 or len(ku) < 3:
        raise AutoAlignError("Too few features found — check the images / channel used for matching.")

    say(f"Matching {len(kx)} × {len(ku)} features…", 0.45)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    good = [m for m, n in (p for p in flann.knnMatch(dxs, dus, k=2) if len(p) == 2)
            if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        raise AutoAlignError(f"Only {len(good)} feature matches survived the ratio test. Check that the IF "
                             f"pane is flipped / rotated to match the Xenium image, and the matching channel.")
    px = np.float32([kx[m.queryIdx].pt for m in good])
    pu = np.float32([ku[m.trainIdx].pt for m in good])

    say("Fitting transform (RANSAC)…", 0.6)
    S, mask = cv2.estimateAffinePartial2D(px, pu, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                          maxIters=5000, confidence=0.999)
    inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), bool)
    n_inl = len(np.unique(np.round(px[inl]), axis=0)) if inl.any() else 0   # distinct Xenium locations
    hint = "Check that the IF pane is flipped / rotated to match the Xenium image, and the matching channel."
    if S is None or n_inl < MIN_INLIERS:
        raise AutoAlignError(f"Only {n_inl} of {len(good)} matches agree on one transform. {hint}")
    # reject degenerate fits (many matches collapsing onto a few points, as happens on noise)
    scale = math.sqrt(abs(np.linalg.det(S[:, :2]))) * xdx / udx      # IF level-0 px per Xenium px
    spread = np.ptp(px[inl], axis=0).max() / max(xg.shape)
    if not (0.02 < scale < 50) or spread < 0.05:
        raise AutoAlignError(f"The matches don't form a plausible transform (scale {scale:.3g} IF px per Xenium px, "
                             f"spread over {spread:.0%} of the image). {hint}")

    # back to level-0 coordinates: Xenium px, raw IF px (undo the display orientation)
    X0 = px[inl].astype(float) * [xdx, xdy]
    U0 = _apply(np.linalg.inv(M_or), pu[inl]) * [udx, udy]
    return {"A": _affine_lstsq(X0, U0), "X": X0, "U": U0,
            "n_features": (len(kx), len(ku)), "n_matches": len(good), "n_inliers": n_inl}


# ----------------------------------------------------------------------------- stage 2: refine
def refine_point(xen, ifim, A, x, if_mode="gray"):
    """Re-locate Xenium point x in the IF by NCC around A(x). Returns (IF raw point, score) or None."""
    cv2 = _cv2()
    half = PATCH + SEARCH
    got = read_region(xen, x[0] - half * REFINE_DS, x[1] - half * REFINE_DS,
                      x[0] + half * REFINE_DS, x[1] + half * REFINE_DS, REFINE_DS)
    if got is None:
        return None
    xr, (ox, oy), (dX, dY) = got
    # template: Xenium patch centred on x
    cx, cy = (x[0] - ox) / dX, (x[1] - oy) / dY
    t0x, t0y = int(round(cx)) - PATCH, int(round(cy)) - PATCH
    if t0x < 0 or t0y < 0:
        return None                                   # too close to the image edge
    tmpl = to_gray(xr)[t0y:t0y + 2 * PATCH + 1, t0x:t0x + 2 * PATCH + 1]
    if tmpl.shape != (2 * PATCH + 1, 2 * PATCH + 1) or tmpl.std() < 1e-3:
        return None
    # search grid: Xenium px of grid pixel (j, i) = g0 + (j, i) * (dX, dY), with g0 so that the
    # template sits at offset (SEARCH, SEARCH) when there is no shift
    g0 = np.array([ox + (t0x - SEARCH) * dX, oy + (t0y - SEARCH) * dY])
    n = 2 * half + 1
    corners = _apply(A, g0 + np.array([[0, 0], [n, 0], [0, n], [n, n]]) * [dX, dY])
    scale = math.sqrt(abs(np.linalg.det(A[:2, :2])))
    got = read_region(ifim, *corners.min(0) - 4 * REFINE_DS * scale,
                      *corners.max(0) + 4 * REFINE_DS * scale, REFINE_DS * scale)
    if got is None:
        return None
    ur, (uox, uoy), (udx, udy) = got
    # grid pixel -> IF region pixel (inverse map for warpAffine)
    G = np.diag([1 / udx, 1 / udy, 1.0]) @ np.array([[1, 0, -uox], [0, 1, -uoy], [0, 0, 1.0]]) @ A @ \
        np.array([[dX, 0, g0[0]], [0, dY, g0[1]], [0, 0, 1.0]])
    warped = cv2.warpAffine(to_gray(ur, if_mode), G[:2], (n, n),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    res = np.nan_to_num(cv2.matchTemplate(warped, tmpl, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0)
    my, mx = np.unravel_index(int(np.argmax(res)), res.shape)
    score = float(res[my, mx])
    if not (0 < mx < res.shape[1] - 1 and 0 < my < res.shape[0] - 1):
        return None                                   # peak at the edge of the search window
    # sub-pixel peak (parabola through the neighbours)
    def sub(c, l, r):
        den = l - 2 * c + r
        return 0.0 if abs(den) < 1e-9 else 0.5 * (l - r) / den
    fx = mx + sub(res[my, mx], res[my, mx - 1], res[my, mx + 1]) - SEARCH
    fy = my + sub(res[my, mx], res[my - 1, mx], res[my + 1, mx]) - SEARCH
    # Xenium content at x is found in the IF where A maps x + shift
    return _apply(A, [x + np.array([fx * dX, fy * dY])])[0], score


FLIP_H = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


def auto_keypoints(xen, ifim, if_orient, if_mode="gray", n_keypoints=12, progress=None):
    """Full pipeline. Returns dict with 'pairs' [(xen_px, if_raw_px)], 'scores', coarse stats and
    'mirrored' (True if it only matched after mirroring the IF relative to if_orient).
    SIFT handles any rotation, so only the mirror state of if_orient matters; if it fails, the
    mirror image is tried too (as InSituPy does)."""
    say = progress or (lambda *a: None)
    try:
        c = coarse_match(xen, ifim, if_orient, if_mode, progress)
        c["mirrored"] = False
    except AutoAlignError as first:
        say("No match in this orientation — trying the mirror image…", 0.6)
        try:
            c = coarse_match(xen, ifim, FLIP_H @ if_orient, if_mode, progress)
            c["mirrored"] = True
        except AutoAlignError:
            raise first
    A = c["A"]
    pairs, scores, tried = [], [], 0
    order = farthest_point_order(c["X"])
    for i in order:
        if len(pairs) >= n_keypoints or tried >= 6 * n_keypoints:
            break
        tried += 1
        say(f"Refining keypoints ({len(pairs)}/{n_keypoints})…", 0.65 + 0.35 * len(pairs) / n_keypoints)
        r = refine_point(xen, ifim, A, c["X"][i], if_mode)
        if r is not None and r[1] >= MIN_SCORE:
            pairs.append((c["X"][i].copy(), r[0]))
            scores.append(r[1])
    if len(pairs) < 3:
        raise AutoAlignError(f"The coarse match worked ({c['n_inliers']} inliers) but only {len(pairs)} keypoints "
                             f"could be refined (tried {tried}). Check the matching channel.")
    c.update(pairs=pairs, scores=scores, tried=tried)
    return c
