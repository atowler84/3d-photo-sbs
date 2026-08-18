"""VR180: the same geometry, wrapped onto a sphere instead of laid on a plane.

A flat side-by-side pair is shown in a headset as a screen hanging in space, and
the player is free to make that screen whatever size it likes.  VR180 is the
other arrangement: the picture is mapped onto the inside of a hemisphere at a
fixed angular scale, so something that subtended thirty degrees to the camera
subtends thirty degrees to the viewer.  That is the whole gain, and it is a real
one -- but it is paid for in periphery, because a photograph does not have any.

**What the container asks for, and what a photograph brings.**  VR180 is 180
degrees across and 180 degrees tall, an equirectangular half-sphere, square per
eye.  A 28mm phone lens covers 65 by 51 degrees, which is 15% of that hemisphere
by solid angle.  The other 85% is not dark because this module gave up on it: it
is dark because nothing was ever pointed at it.  `render` hands back the mask of
what is real so the caller can say so out loud, and fades the edge rather than
cutting it, on the grounds that an honest absence reads better in a headset than
a hard-edged rectangle floating in a void.

**Where the depth comes from.**  The depth model runs on the photograph, before
any of this.  It is trained on ordinary perspective images and an equirectangular
one is not that -- straight lines are not straight in it, and the network has
never seen a picture where they were not.  So the colour and the inverse depth
are estimated flat and warped across together, and the stereo is rendered
afterwards, in equirectangular space.

**Why the stereo is not simply the flat renderer pointed at a warped picture.**
One fixed pair of eyes cannot see a whole hemisphere.  Turn to look ninety
degrees left and eyes that were side by side are now one behind the other, with
no separation left to give.  The standard answer is omnidirectional stereo, where
every column is taken from a different point on a circle the size of the eye
separation, so the pair is always across the line of sight.  It is not a
physically consistent single viewpoint and it cannot be -- but it is what every
VR180 camera records and every player expects, and the disparity it asks for is

    dtheta = B * (1/Z - 1/Zc)

radians, which is `stereo.half_disparity` exactly, with the projection's
pixels-per-radian standing in where the focal length used to be.  The one
addition is a cosine taper towards the poles, where the separation has to fall
to nothing: looking straight up there is no "across the line of sight" left to
put two eyes on, and insisting on some anyway is what makes badly made VR180
hurt to look at.

**No metadata is written here.**  A finished file still needs an `st3d`/`sv3d`
box pair (video) or GPano XMP (stills) before a player will recognise the
projection without being told.  Until that exists, set the player to
equirectangular 180, side-by-side, by hand.
"""

import math

import torch

from . import stereo

# Fixed by the format: 180 degrees each way, which makes each eye square.
FOV = 180.0
# Near-to-far angular separation `auto` aims for, in degrees.  The flat
# renderer aims at a percentage of frame width, and that number does not survive
# the move: 2% of a 180-degree frame is 3.6 degrees of parallax, several times
# what anyone can fuse.  This projection thinks in angles, so it asks in angles.
TARGET_DEG = 0.6
# The ceiling on separation, for the same reason `Settings.limit_pct` exists and
# in the units that mean something here.  Beyond about a degree the two images
# stop fusing and start doubling.
LIMIT_DEG = 1.2
# How wide the fade at the edge of the real picture is.
FALLOFF_DEG = 4.0
# Ceilings on the per-eye square.  A still can afford a big one; a clip has to
# come back out of a hardware decoder, and 2048 per eye is already 4096 across.
MAX_SIZE = 4096
VIDEO_MAX_SIZE = 2048
# Output pixels per band, so peak memory stops tracking the whole equirectangular
# frame -- the same trick `stereo._render_eye` plays, for the same reason.
_BAND_PIXELS = 8_000_000


def even(n):
    """Encoders refuse odd dimensions, and a square that is even stays even
    when it is doubled into a pair."""
    return max(2, int(n) - int(n) % 2)


def hfov(focal_px, width):
    """The horizontal field of view of a lens, in degrees."""
    return math.degrees(2.0 * math.atan(width / (2.0 * float(focal_px))))


def per_radian(size):
    """Pixels per radian of the finished projection, which is what stands in for
    the focal length once the picture is on a sphere."""
    return size / math.radians(FOV)


def per_eye(width, focal_px, cap=MAX_SIZE):
    """The square to render each eye at.

    The size that would keep the photograph's own detail is the one that gives
    the sphere the same pixels per degree the photograph had -- which for a
    65-degree lens means a square nearly three times the width of the original,
    and for anything off a modern camera lands past 10000 a side.  So it is
    asked for and then capped, and the caller is left to report the shortfall
    rather than have it happen quietly.
    """
    natural = width * FOV / hfov(focal_px, width)
    return even(max(64, min(cap, round(natural))))


def auto_target():
    """`TARGET_DEG` restated as the percentage of frame width that
    `stereo.auto_geometry` takes, so that one piece of scene-fitting logic
    serves both projections instead of two that can drift apart."""
    return 100.0 * TARGET_DEG / FOV


def _elevation(top, rows, size, device, dtype):
    """Elevation of each output row, in radians: +90 at the top, -90 at the
    bottom, which is the way round equirectangular has it."""
    row = torch.arange(top, top + rows, device=device, dtype=dtype)
    return math.radians(FOV) / 2.0 - (row + 0.5) / size * math.radians(FOV)


def _grid(top, rows, size, focal_px, src_h, src_w, device, dtype):
    """Where each equirectangular pixel in a band of rows reads from in the
    source photograph, and whether it reads from anywhere at all.

    Azimuth and elevation give a direction; the direction is put back through
    the pinhole the depth model reported.  Anything behind the camera, or past
    the edge of the frame, is not in the photograph and is marked as such.
    """
    col = torch.arange(size, device=device, dtype=dtype)
    az = (col + 0.5) / size * math.radians(FOV) - math.radians(FOV) / 2.0
    el = _elevation(top, rows, size, device, dtype)[:, None]

    cos_el, sin_el = torch.cos(el), torch.sin(el)
    x = torch.sin(az)[None, :] * cos_el
    y = sin_el.expand(rows, size)
    z = torch.cos(az)[None, :] * cos_el

    # Behind the camera divides by a negative and folds the picture back on
    # itself, so it is held at 1 for the arithmetic and thrown away by the mask.
    front = z > 1e-6
    depth = torch.where(front, z, torch.ones_like(z))
    u = (src_w - 1) / 2.0 + focal_px * x / depth
    v = (src_h - 1) / 2.0 - focal_px * y / depth
    valid = front & (u >= 0) & (u <= src_w - 1) & (v >= 0) & (v <= src_h - 1)

    # grid_sample's normalised coordinates, align_corners=False: a pixel centre
    # at i sits at (2i + 1) / n - 1.
    gx = (2.0 * u + 1.0) / src_w - 1.0
    gy = (2.0 * v + 1.0) / src_h - 1.0
    return torch.stack((gx, gy), dim=-1)[None], valid


def project(source, focal_px, size):
    """Warp a `[C, H, W]` perspective image onto a `[C, size, size]` equirectangular
    half-sphere, and say which of its pixels came from anywhere.

    Banded over output rows: an 8192-square projection of a large photograph is
    several gigabytes if it is built in one piece, and nothing here needs it to
    be.
    """
    import torch.nn.functional as F

    src_h, src_w = source.shape[-2:]
    out = torch.zeros(source.shape[0], size, size, device=source.device, dtype=source.dtype)
    mask = torch.zeros(size, size, device=source.device, dtype=torch.bool)
    band = max(1, _BAND_PIXELS // max(size, 1))
    for top in range(0, size, band):
        rows = min(band, size - top)
        grid, valid = _grid(top, rows, size, focal_px, src_h, src_w,
                            source.device, source.dtype)
        out[:, top:top + rows] = F.grid_sample(
            source[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        mask[top:top + rows] = valid
    return out, mask


def coverage(mask):
    """The share of the hemisphere that is photograph, by solid angle.

    Deliberately not the share of the pixels.  Equirectangular packs far more
    pixels into a degree near the pole than at the equator, so counting them
    would flatter or libel the result depending on nothing but where in the
    frame the picture happened to land.  What a viewer notices is how much of
    what they can turn to look at is real, and that is solid angle: the same
    quantity that says a 28mm lens brings 15% of a hemisphere with it.
    """
    weight = torch.cos(_elevation(0, mask.shape[0], mask.shape[0],
                                  mask.device, torch.float32))[:, None]
    return float((mask.float() * weight).sum() / weight.expand_as(mask).sum())


def _falloff(mask, size):
    """A soft edge at the boundary of the real picture, in place of a cut one.

    The blur of a hard mask runs from one deep inside to zero outside, which is
    the ramp wanted; multiplying by the mask again keeps the fade strictly
    inside, so no pixel that came from nowhere is ever shown at any strength.
    """
    radius = max(1, round(FALLOFF_DEG * size / FOV))
    soft = stereo._box(mask.float()[None, None], radius)[0, 0].clamp_(0, 1)
    return mask.float() * (soft * soft * (3.0 - 2.0 * soft))  # smoothstep


def half_disparity(inverse, size, eyes_mm, focus_m, limit_deg=LIMIT_DEG, elevation=None):
    """Half the omnidirectional-stereo separation, in pixels, for every pixel.

    The formula is `stereo.half_disparity`'s, because it is the same geometry:
    what changes is that a radian of angle rather than a pixel of sensor is what
    the separation is measured against, so pixels-per-radian goes in where the
    focal length was.  The clamp is in degrees for the same reason -- a
    percentage of the frame width means a percentage of 180 degrees here, which
    is not a quantity anybody's comfort is described in.
    """
    ppr = per_radian(size)
    half = stereo.half_disparity(inverse, ppr, eyes_mm, focus_m)
    cap = math.radians(limit_deg) * ppr / 2.0
    half = half.clamp(-cap, cap)
    if elevation is None:
        elevation = _elevation(0, size, size, inverse.device, torch.float32)
    # Towards the poles the two eyes line up along the view direction and the
    # separation has to go to nothing.  Without this the top and bottom of the
    # sphere ask for parallax that no arrangement of two eyes could produce.
    return half * torch.cos(elevation.to(half.dtype))[:, None]


def render(rgb, inverse, focal_px, eyes_mm, focus_m, size, limit_deg=LIMIT_DEG):
    """One flat frame and its depth in; a VR180 eye pair and its coverage out.

    `rgb` is `[3, H, W]` in [0, 1] and `inverse` the matching `[H, W]` inverse
    depth in 1/metres -- both as the flat path has them, because both are warped
    from there rather than estimated here.
    """
    equirect, mask = project(rgb, focal_px, size)
    depth, _ = project(inverse[None], focal_px, size)
    # Nothing was ever pointed at the void, so it has no depth either.  Parked at
    # the screen plane it asks for no separation, which keeps it still and stops
    # the splat dragging it sideways over the edge of the real picture.
    depth = torch.where(mask, depth[0], torch.full_like(depth[0], 1.0 / max(focus_m, 1e-6)))

    half = half_disparity(depth, size, eyes_mm, focus_m, limit_deg)
    # No margin: the flat path trims the sliver at each edge that only one eye
    # reaches, and here that sliver is angle.  Trimming it would quietly narrow
    # the field of view and put every remaining pixel at the wrong bearing.
    left, right = stereo.make_pair(equirect, half, margin=0)

    fade = _falloff(mask, size)
    return left * fade, right * fade, mask
