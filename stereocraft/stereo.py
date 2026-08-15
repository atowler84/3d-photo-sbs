"""Depth-image-based rendering: one photo + one depth map -> a stereo pair.

The depth map arrives in metres, so the separation between the eyes is not a
number to be tuned but one to be worked out: `half_disparity` places two eyes a
real distance apart, points them at a real focus distance, and computes the
parallax that geometry gives.  What the settings choose is where the viewer is
standing, not how much depth to draw.

Human eyes are about 65mm apart and the world is metres away, so the separation
that falls out is a couple of percent of the frame -- which makes the regions
each eye uncovers only a few pixels wide.  That is what lets the render skip a
full layered-depth-image reconstruction: a z-buffered forward splat of the
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


def percentiles(depth, low=2.0, high=98.0):
    """The raw depth values the `low`th and `high`th percentiles sit at.

    Kept apart from applying them because a video wants to smooth the pair over
    time before using it: re-derived from scratch every frame, the range twitches
    as the scene moves and drags the whole depth map with it.
    """
    flat = depth.flatten()
    if flat.numel() > 1 << 20:  # torch.quantile has an input-size ceiling
        flat = flat[:: flat.numel() // (1 << 20) + 1]
    lo, hi = torch.quantile(flat.float(), torch.tensor([low / 100, high / 100], device=flat.device))
    return lo, hi


def apply_range(depth, lo, hi):
    """Map `[lo, hi]` onto [0, 1], with everything outside it clamped."""
    if hi - lo < 1e-6:
        return torch.zeros_like(depth)
    return ((depth - lo) / (hi - lo)).clamp_(0, 1)


def normalize(depth, low=2.0, high=98.0):
    """Map raw relative depth onto [0, 1] using percentiles, so a few stray
    pixels (a blown-out sky, a speck of lens flare) cannot squash the range."""
    return apply_range(depth, *percentiles(depth, low, high))


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


def max_margin(width, limit=3.0):
    """The widest trim the comfort clamp can ever call for, at this width.

    A still is trimmed by whatever its own disparity turned out to need.  A clip
    cannot be: frames trimmed by different amounts come out different sizes, and
    no encoder will take that.  The clamp puts a hard ceiling on disparity, so
    that ceiling is a margin every frame of a clip can share.
    """
    return min(int(math.ceil(limit / 100.0 * width / 2.0)), (width - 1) // 2)


def half_disparity(inverse, focal, eyes_mm=65.0, focus_m=3.0, limit=3.0, width=None):
    """Half the separation between the eyes, in pixels, for every pixel.

    This is the whole of the geometry.  Two eyes `eyes_mm` apart, both looking at
    a screen placed `focus_m` away, see a point at distance Z separated by

        d = f * B * (1/Z - 1/Zc)

    -- the shifted-sensor arrangement, which is the one that does not introduce
    the vertical parallax a toed-in pair would.  Points at the focus distance land
    on the screen with no separation at all, nearer ones come forward, and
    everything beyond it recedes towards a *finite* limit of `f * B / Zc` rather
    than being stretched out by however the depth map happened to be scaled.
    That finite limit is the difference between this and guessing.

    `focal` is in pixels at the width being rendered, and `inverse` is in
    1/metres, which is what the depth backends hand over.

    The clamp exists because metric depth is honest about how close things get: a
    hand held out at 300 mm really does subtend a parallax no one can fuse, and
    the physics will happily say so.  `limit` caps it as a percentage of frame
    width, which is the units comfort is usually discussed in.
    """
    half = focal * (eyes_mm / 1000.0) * (inverse - 1.0 / focus_m) / 2.0
    if width:
        cap = limit / 100.0 * width / 2.0
        half = half.clamp(-cap, cap)
    return half


def auto_geometry(inverse, width, focal, target=2.0, forward=0.12):
    """Choose the eye separation and screen plane to suit this scene.

    Physically literal eyes are the wrong default, and measuring said so: 65mm
    at a subject 300mm away asks for 16% of the frame width, five times past what
    anyone can fuse, while the same eyes on a 20m telephoto shot ask for 0.3% and
    render something almost flat.  Both are *correct* -- that really is what a
    person standing at the camera would see -- but the person is not standing at
    the camera, they are looking at a screen.

    So the separation is chosen per scene, which is what a stereographer does
    rather than a mistake to apologise for: a wider baseline than human for a
    distant landscape, a narrower one for a close-up.  What matters is that only
    the two free parameters move.  The *shape* of the depth stays exactly what
    the metric geometry says -- parallax falling off as 1/Z, distant things
    converging to a finite separation -- and that shape is the whole realism
    gain.  Scaling it is a choice about how big the viewer should feel, not an
    approximation of the geometry.

    `target` is the near-to-far separation to aim for as a percentage of frame
    width, and `forward` the share of the picture left in front of the window.
    """
    flat = inverse.flatten().float()
    if flat.numel() > 1 << 20:  # torch.quantile has an input-size ceiling
        flat = flat[:: flat.numel() // (1 << 20) + 1]
    # Percentiles, not the extremes, so one stray pixel cannot set the baseline.
    far, near, focus = torch.quantile(flat, torch.tensor([0.02, 0.98, 1.0 - forward],
                                                         device=flat.device))
    span = float(near - far)
    if span < 1e-9:  # a scene at one distance has no stereo to give
        return 0.0, max(float(1.0 / focus.clamp_min(1e-6)), 0.1)
    eyes_mm = 1000.0 * (target / 100.0 * width) / (focal * span)
    return eyes_mm, max(float(1.0 / focus.clamp_min(1e-6)), 0.05)


def make_pair(image, half, margin=None):
    """Turn one image and its disparity field into a left/right pair.

    `image` is a float [3,H,W] tensor in [0,1]; `half` is the signed
    half-disparity in pixels from `half_disparity`, positive towards the viewer.
    `margin` overrides how much is trimmed off each end, for a caller that needs
    every frame the same size; see `max_margin`.
    """
    width = image.shape[2]
    half = half.to(image.dtype)

    left = _render_eye(image, half, +1)
    right = _render_eye(image, half, -1)

    # Both eyes shift content sideways, leaving a sliver at the frame edge that no
    # real pixel reaches.  Trimming it beats filling it, and costs ~1% of width.
    if margin is None:
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
