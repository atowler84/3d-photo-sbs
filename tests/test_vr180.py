"""The spherical geometry, which has to be exactly as right as the flat one.

A projection that is wrong by a few degrees still looks like a picture -- it
just sits at the wrong bearing, and the only symptom is that a headset feels
subtly wrong to be inside.  So the angles are checked against the arithmetic
rather than against how the result looks.
"""

import math

import pytest
import torch

from stereocraft import stereo, vr180


def inv(*metres):
    return torch.tensor([[1.0 / z for z in metres]])


def dot(size=64, at=None, value=1.0):
    """A source photo that is dark everywhere but one pixel."""
    image = torch.zeros(1, size, size)
    row, col = at or (size // 2, size // 2)
    image[0, row, col] = value
    return image


class TestLens:
    def test_hfov_of_a_28mm_phone(self):
        # 36mm of film across a 28mm lens, which is what most phones point at
        # the world -- and the number every coverage figure here rests on.
        assert vr180.hfov(1000.0 * 28 / 36, 1000) == pytest.approx(65.47, abs=0.01)

    def test_a_longer_lens_sees_less(self):
        assert vr180.hfov(2000.0, 1000) < vr180.hfov(1000.0, 1000)

    def test_per_radian_spreads_the_square_over_180_degrees(self):
        assert vr180.per_radian(1800) == pytest.approx(1800 / math.pi)


class TestPerEye:
    """Asking for the photo's own detail, and settling for what fits."""

    def test_asks_for_the_angular_density_the_photo_had(self):
        width, focal = 1000, 1000.0 * 28 / 36  # 65.47 degrees
        assert vr180.per_eye(width, focal, cap=100000) == pytest.approx(
            vr180.even(round(width * 180 / 65.47)), abs=2)

    def test_the_cap_is_a_cap(self):
        assert vr180.per_eye(8000, 1000.0, cap=4096) == 4096

    def test_always_even(self):
        assert vr180.per_eye(999, 777.0, cap=1001) % 2 == 0


class TestProjection:
    """Where each equirectangular pixel reads from in the photograph."""

    def test_the_centre_of_the_sphere_is_the_centre_of_the_photo(self):
        # Odd sizes throughout, so that "the centre" is a pixel rather than the
        # crack between two of them.
        out, mask = vr180.project(dot(65), focal_px=65.0, size=33)
        assert bool(mask[16, 16])
        assert divmod(int(torch.argmax(out[0])), 33) == (16, 16)

    @pytest.mark.parametrize("col", [180, 200, 240, 255])
    def test_an_azimuth_lands_where_the_tangent_says(self, col):
        """A ray at theta off-axis meets the sensor at f*tan(theta), which is the
        whole of what "perspective" means and the one thing worth checking."""
        size, focal, src = 360, 500.0, 800
        grid, valid = vr180._grid(size // 2, 1, size, focal, src, src,
                                  torch.device("cpu"), torch.float32)
        azimuth = math.radians((col + 0.5) / size * vr180.FOV - vr180.FOV / 2)
        gx = float(grid[0, 0, col, 0])
        u = ((gx + 1.0) * src - 1.0) / 2.0  # back out of grid_sample's units
        assert u == pytest.approx((src - 1) / 2 + focal * math.tan(azimuth), abs=0.01)
        assert bool(valid[0, col])

    def test_the_mask_stops_where_the_lens_did(self):
        """An 800-wide frame at f=500 sees 77.3 degrees, so 38.6 either side --
        and one column past that is not in the photograph however well the
        projection arithmetic behaves there."""
        size, focal, src = 360, 500.0, 800
        _, valid = vr180._grid(size // 2, 1, size, focal, src, src,
                               torch.device("cpu"), torch.float32)
        edge = math.degrees(math.atan((src - 1) / 2 / focal))
        inside = int((edge + 90.0) / 180.0 * size - 0.5)
        assert bool(valid[0, inside]), "the last column the lens reached"
        assert not bool(valid[0, inside + 1]), "the first one it did not"

    def test_nothing_behind_the_camera_is_ever_valid(self):
        _, mask = vr180.project(torch.ones(1, 64, 64), focal_px=64.0, size=64)
        assert not bool(mask[:, 0].any()), "the left pole looks backwards"
        assert not bool(mask[:, -1].any()), "and so does the right one"

    @pytest.mark.parametrize("focal", [128.0, 256.0, 512.0])
    def test_coverage_is_the_share_of_the_hemisphere_the_lens_saw(self, focal):
        """The solid angle a rectilinear frame covers is 4*asin(sin(a/2)sin(b/2)),
        and a hemisphere is 2pi of them.  Everything said about how much of a
        VR180 frame is real rests on this agreeing."""
        size, src = 512, 256
        _, mask = vr180.project(torch.ones(1, src, src), focal, size)
        a = b = math.radians(vr180.hfov(focal, src))
        expected = 4 * math.asin(math.sin(a / 2) * math.sin(b / 2)) / (2 * math.pi)
        assert vr180.coverage(mask) == pytest.approx(expected, rel=0.02)

    def test_a_28mm_phone_brings_about_a_seventh_of_a_hemisphere(self):
        """The number the whole feature has to be judged against."""
        src_w, src_h = 400, 300
        _, mask = vr180.project(torch.ones(1, src_h, src_w), 400.0 * 28 / 36, 512)
        assert vr180.coverage(mask) == pytest.approx(0.151, abs=0.01)

    def test_counting_pixels_would_have_said_something_else(self):
        """Equirectangular crowds pixels towards the poles, so the share of the
        frame that is lit is not the share of the sphere that is real."""
        _, mask = vr180.project(torch.ones(1, 256, 256), 256.0, 512)
        assert float(mask.float().mean()) < vr180.coverage(mask) * 0.8

    def test_a_narrow_lens_covers_less_than_a_wide_one(self):
        _, wide = vr180.project(torch.ones(1, 128, 128), 64.0, 128)
        _, narrow = vr180.project(torch.ones(1, 128, 128), 256.0, 128)
        assert float(narrow.float().mean()) < float(wide.float().mean())


class TestOdsDisparity:
    """dtheta = B*(1/Z - 1/Zc) radians, tapering to nothing at the poles."""

    def test_matches_the_formula_at_the_equator(self):
        size, z, eyes, focus = 360, 5.0, 65.0, 3.0
        half = vr180.half_disparity(inv(z), size, eyes, focus, elevation=torch.zeros(1))
        radians = float(half) * 2 / vr180.per_radian(size)
        assert radians == pytest.approx(0.065 * (1 / z - 1 / focus), abs=1e-6)

    def test_zero_at_the_screen_plane(self):
        half = vr180.half_disparity(inv(3.0), 360, 65.0, 3.0, elevation=torch.zeros(1))
        assert float(half) == pytest.approx(0.0, abs=1e-9)

    def test_the_poles_have_no_separation_to_give(self):
        """Looking straight up there is no across-the-line-of-sight left to put
        two eyes on, and a projection that asks for parallax there is asking for
        something no pair of eyes could produce."""
        depth = inv(1.0).expand(3, 1)
        poles = torch.tensor([math.pi / 2, 0.0, -math.pi / 2])
        half = vr180.half_disparity(depth, 360, 65.0, 3.0, elevation=poles)
        up, level, down = half[:, 0].tolist()
        assert up == pytest.approx(0.0, abs=1e-6)
        assert down == pytest.approx(0.0, abs=1e-6)
        assert abs(level) > 0

    def test_the_taper_is_a_cosine(self):
        depth = inv(1.0).expand(2, 1)
        at = torch.tensor([0.0, math.pi / 3])  # 60 degrees up, so half the effect
        half = vr180.half_disparity(depth, 360, 65.0, 3.0, elevation=at)
        assert float(half[1, 0]) == pytest.approx(float(half[0, 0]) * 0.5, rel=1e-4)

    def test_the_clamp_is_in_degrees(self):
        size, limit = 360, 1.2
        half = vr180.half_disparity(inv(0.02), size, 500.0, 3.0, limit_deg=limit,
                                    elevation=torch.zeros(1))
        assert float(half.abs()) <= math.radians(limit) * vr180.per_radian(size) / 2 + 1e-6

    def test_defaults_to_one_elevation_per_row(self):
        half = vr180.half_disparity(torch.full((8, 8), 0.5), 8, 65.0, 3.0)
        assert half.shape == (8, 8)


class TestRender:
    def test_each_eye_is_the_square_that_was_asked_for(self):
        rgb = torch.rand(3, 48, 64)
        left, right, mask = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, size=32)
        assert left.shape == right.shape == (3, 32, 32)
        assert mask.shape == (32, 32)

    def test_nothing_is_trimmed(self):
        """The flat path trims the sliver only one eye reaches.  Here that sliver
        is angle, and trimming it would put every remaining pixel at the wrong
        bearing while looking perfectly fine."""
        rgb = torch.rand(3, 48, 64)
        left, _, _ = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, size=40)
        assert left.shape[2] == 40

    def test_the_void_stays_dark(self):
        rgb = torch.ones(3, 64, 64)
        left, right, mask = vr180.render(rgb, torch.full((64, 64), 0.5), 64.0, 65.0, 3.0, size=64)
        assert float(left[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)
        assert float(right[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)

    def test_the_edge_is_faded_rather_than_cut(self):
        """A hard rectangle floating in a void is the thing this is avoiding, so
        somewhere inside the real picture there has to be a partial pixel."""
        rgb = torch.ones(3, 64, 64)
        left, _, mask = vr180.render(rgb, torch.full((64, 64), 0.5), 64.0, 65.0, 3.0, size=96)
        inside = left[0][mask]
        assert float(inside.max()) == pytest.approx(1.0, abs=1e-3), "the middle is untouched"
        assert bool((inside < 0.9).any()), "and the rim is on its way down"

    def test_a_flat_scene_leaves_the_two_eyes_alike(self):
        """Everything at the screen plane has no separation, so the pair matches."""
        rgb = torch.rand(3, 64, 64)
        flat = torch.full((64, 64), 1.0 / 3.0)  # every pixel at the focus distance
        left, right, _ = vr180.render(rgb, flat, 64.0, 65.0, 3.0, size=48)
        assert torch.allclose(left, right, atol=1e-5)

    def test_depth_moves_the_eyes_apart(self):
        rgb = torch.rand(3, 64, 64)
        near = torch.full((64, 64), 1.0 / 0.8)
        left, right, _ = vr180.render(rgb, near, 64.0, 200.0, 3.0, size=48)
        assert not torch.allclose(left, right, atol=1e-3)


class TestAutoTarget:
    def test_restates_the_angle_as_the_flat_path_s_percentage(self):
        """`auto_geometry` is shared, so the angle has to arrive in its units."""
        assert vr180.auto_target() == pytest.approx(100.0 * vr180.TARGET_DEG / 180.0)

    def test_the_shared_geometry_hits_that_angle(self):
        """Feed `auto_geometry` the projection's pixels-per-radian and it should
        pick a baseline that spreads the scene over `TARGET_DEG` degrees."""
        size = 360
        scene = 1.0 / torch.linspace(2.0, 40.0, 4096)[None]
        eyes, _ = stereo.auto_geometry(scene, size, vr180.per_radian(size),
                                       vr180.auto_target())
        # The same percentiles auto_geometry works from, so this checks the units
        # rather than re-deriving the statistics.
        lo, hi = torch.quantile(scene.flatten(), torch.tensor([0.02, 0.98]))
        assert (eyes / 1000.0) * float(hi - lo) == pytest.approx(
            math.radians(vr180.TARGET_DEG), rel=1e-3)
