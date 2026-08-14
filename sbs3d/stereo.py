"""Depth-image-based rendering: one photo + one depth map -> a stereo pair.

The baseline for a 3D photo is tiny (a couple of percent of the frame width), so
the regions each eye uncovers are only a few pixels wide.  That makes a full
layered-depth-image reconstruction unnecessary: a z-buffered forward splat of the
disparity followed by a sub-pixel backward resample gives the same geometry at
native photo resolution, in milliseconds instead of minutes.
"""

import math

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")
# How far the background is stretched back across a gap when filling it.  0 would
# smear one column of pixels into a flat band; 1 would mirror the background at
# full scale and show the fold.  Half-scale keeps the texture without the seam.
_STRETCH = 0.5
# Rows are independent, so tall images render in bands of roughly this many
# pixels to keep peak memory flat no matter how big the photo is.
_BAND_PIXELS = 8_000_000


def _box(x, r):
    """Separable box blur with replicated edges."""
    k = 2 * r + 1
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.avg_pool2d(x, (1, k), stride=1)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.avg_pool2d(x, (k, 1), stride=1)


def guided_upsample(depth, guide, radius=4, eps=1e-3):
    """Joint upsample `depth` to `guide`'s resolution with a guided filter.

    Coefficients are solved on the small grid and applied on the large one, so the
    cost is set by the depth map, not the photo.  Edges end up aligned with the
    photo instead of the blurry ramps bilinear interpolation would leave.
    """
    h, w = guide.shape[-2:]
    guide_lo = F.interpolate(guide, size=depth.shape[-2:], mode="area")

    mean_i = _box(guide_lo, radius)
    mean_p = _box(depth, radius)
    var_i = _box(guide_lo * guide_lo, radius) - mean_i * mean_i
    cov_ip = _box(guide_lo * depth, radius) - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    a = F.interpolate(_box(a, radius), size=(h, w), mode="bilinear", align_corners=False)
    b = F.interpolate(_box(b, radius), size=(h, w), mode="bilinear", align_corners=False)
    return a * guide + b


def normalize(depth, low=2.0, high=98.0):
    """Map raw relative depth onto [0, 1] using percentiles, so a few stray
    pixels (a blown-out sky, a speck of lens flare) cannot squash the range."""
    flat = depth.flatten()
    if flat.numel() > 1 << 20:  # torch.quantile has an input-size ceiling
        flat = flat[:: flat.numel() // (1 << 20) + 1]
    lo, hi = torch.quantile(flat.float(), torch.tensor([low / 100, high / 100], device=flat.device))
    if hi - lo < 1e-6:
        return torch.zeros_like(depth)
    return ((depth - lo) / (hi - lo)).clamp_(0, 1)


def _sample_columns(image, x):
    """Bilinear gather along each row at floating-point column positions `x`.

    Done by hand rather than with grid_sample so a pixel only ever mixes with its
    horizontal neighbours -- no vertical coordinate is involved, which keeps the
    result exact and independent of how the frame is banded.
    """
    width = image.shape[-1]
    x = x.clamp(0, width - 1)
    left = x.floor()
    weight = (x - left).to(image.dtype)
    left = left.long()
    right = (left + 1).clamp(max=width - 1)
    columns = image.shape[0]
    lo = image.gather(2, left[None].expand(columns, -1, -1))
    hi = image.gather(2, right[None].expand(columns, -1, -1))
    return torch.lerp(lo, hi, weight[None])


def _render_eye(image, half, sign, band=None):
    """Render one eye.  `half` is the signed half-disparity in pixels; `sign` is
    +1 for the left eye (near objects swing right) and -1 for the right eye.

    Every step below works strictly within a row, so wide photos are rendered in
    horizontal bands: the result is identical and peak memory stops tracking the
    full frame.
    """
    height, width = half.shape
    band = band or max(1, _BAND_PIXELS // max(width, 1))
    if band >= height:
        return _render_band(image, half, sign)
    out = torch.empty_like(image)
    for top in range(0, height, band):
        bottom = min(top + band, height)
        out[:, top:bottom] = _render_band(image[:, top:bottom], half[top:bottom], sign)
    return out


def _render_band(image, half, sign):
    _, height, width = image.shape
    device = image.device
    xs = torch.arange(width, device=device, dtype=image.dtype).view(1, width).expand(height, width)

    # --- forward splat with a z-buffer -------------------------------------
    # Every source pixel votes for the column it lands in; the largest half-
    # disparity wins, which is exactly "the nearest surface occludes".
    target_x = xs + sign * half
    rows = (torch.arange(height, device=device) * width).view(height, 1)
    buf = torch.full((height * width,), NEG_INF, device=device, dtype=image.dtype)
    floor = torch.floor(target_x).long()
    for step in (0, 1):  # two-tap splat closes the 1px cracks a stretch opens up
        col = floor + step
        ok = (col >= 0) & (col < width)
        buf.scatter_reduce_(0, (rows + col)[ok], half[ok], reduce="amax", include_self=True)
    buf = buf.view(height, width)
    valid = torch.isfinite(buf)

    # --- backward resample --------------------------------------------------
    # A pixel that landed at `x` came from `x - sign * half`, and sampling there
    # with bilinear weights recovers sub-pixel detail the splat rounded away.
    out = _sample_columns(image, xs - sign * torch.where(valid, buf, torch.zeros_like(buf)))
    if bool(valid.all()):
        return out

    # --- fill what nothing landed on ---------------------------------------
    # A gap is background that the foreground used to hide, so it has to be
    # filled from whichever side is *further away*; taking the nearer side would
    # drag a ghost of the foreground edge into the gap.
    cols = torch.arange(width, device=device)
    left_idx = torch.cummax(torch.where(valid, cols, torch.full_like(cols, -1)), dim=1).values
    right_idx = torch.flip(
        torch.cummin(torch.flip(torch.where(valid, cols, torch.full_like(cols, width)), (1,)), dim=1).values, (1,)
    )
    has_left, has_right = left_idx >= 0, right_idx < width
    depth_left = buf.gather(1, left_idx.clamp(min=0))
    depth_right = buf.gather(1, right_idx.clamp(max=width - 1))
    take_left = has_left & (~has_right | (depth_left <= depth_right))
    anchor = torch.where(take_left, left_idx.clamp(min=0), right_idx.clamp(max=width - 1))

    # Flood the gaps from the background edge first, so the stretch below always
    # reads a defined pixel, then replace them with a gently stretched, mirrored
    # copy of that background: same texture, no flat smear, no foreground echo.
    base = torch.where(valid[None], out, out.gather(2, anchor[None].expand_as(out)))
    stretched = _sample_columns(base, anchor - _STRETCH * (xs - anchor))
    return torch.where(valid[None], out, stretched)


def make_pair(image, depth, disparity=2.0, convergence=0.9):
    """Turn one image into a left/right pair.

    `image` is a float [3,H,W] tensor in [0,1], `depth` a [H,W] tensor in [0,1]
    where 1 is nearest.  `disparity` is the total near-to-far separation as a
    percentage of image width, and `convergence` picks the depth that sits on the
    screen plane -- everything nearer pops out, everything further recedes.
    """
    _, _, width = image.shape
    span = disparity / 100.0 * width
    half = (span * (depth - convergence) / 2.0).to(image.dtype)

    left = _render_eye(image, half, +1)
    right = _render_eye(image, half, -1)

    # Both eyes shift content sideways, leaving a sliver at the frame edge that no
    # real pixel reaches.  Trimming it beats filling it, and costs ~1% of width.
    margin = min(int(math.ceil(float(half.abs().max()))), (width - 1) // 2)
    if margin > 0:
        left = left[:, :, margin:-margin]
        right = right[:, :, margin:-margin]
    return left, right


def compose(left, right, cross_eyed=False):
    """Glue the pair into one frame: left|right for headsets and parallel
    free-viewing, right|left for the cross-eyed method."""
    if cross_eyed:
        left, right = right, left
    return torch.cat([left, right], dim=2)
