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


def lens(equivalent_mm, width):
    """Focal length in pixels for a 35mm-equivalent lens at this width."""
    return float(equivalent_mm) / 36.0 * width


def dot(size=64, at=None):
    """A source photo that is dark everywhere but one pixel."""
    image = torch.zeros(1, size, size)
    row, col = at or (size // 2, size // 2)
    image[0, row, col] = 1.0
    return image


class TestLens:
    def test_the_fov_of_a_28mm_phone(self):
        assert vr180.fov(lens(28, 1000), 1000) == pytest.approx(65.47, abs=0.01)

    def test_it_works_on_either_axis(self):
        """A 3:2 frame is taller in pixels than it is in nothing else, and the
        vertical angle falls out of the same arithmetic."""
        focal = lens(28, 3000)
        assert vr180.fov(focal, 2000) < vr180.fov(focal, 3000)

    def test_a_longer_lens_sees_less(self):
        assert vr180.fov(2000.0, 1000) < vr180.fov(1000.0, 1000)


class TestPatch:
    """Where the stored picture sits in the 360-by-180 frame it is a piece of."""

    def test_a_lens_covers_exactly_its_own_angles(self):
        """The azimuth a column lands at is atan(u/f) and does not depend on the
        row, so the patch needs no searching for -- it is the lens."""
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.span_az == pytest.approx(vr180.fov(lens(28, 4000), 4000))
        assert spot.span_el == pytest.approx(vr180.fov(lens(28, 4000), 3000))

    def test_the_full_frame_it_belongs_to_is_two_to_one(self):
        """Equirectangular is 360 across and 180 down.  If the stored piece does
        not imply that shape, every offset recorded against it is wrong."""
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.full_width == pytest.approx(2 * spot.full_height, rel=0.01)

    def test_the_piece_sits_in_the_middle(self):
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.left == pytest.approx((spot.full_width - spot.width) / 2, abs=1)
        assert spot.top == pytest.approx((spot.full_height - spot.height) / 2, abs=1)

    def test_a_full_frame_is_the_square_the_format_asks_for(self):
        spot = vr180.patch(lens(28, 1000), 1000, 750, full=True)
        assert spot.span_az == spot.span_el == 180.0
        assert spot.width == spot.height
        assert not spot.cropped

    def test_a_full_frame_is_half_the_sphere_across(self):
        """180 of a possible 360, so a quarter falls off each side -- which is
        the whole of what makes a VR180 file VR180 rather than 360."""
        spot = vr180.patch(lens(28, 1000), 1000, 750, full=True)
        assert spot.full_width == 2 * spot.width
        assert spot.full_height == spot.height

    def test_cropping_keeps_the_source_s_own_width(self):
        """The point of the whole exercise: stored at its own width the picture
        keeps its own detail, where the square gave it whatever share of itself
        its lens had earned."""
        spot = vr180.patch(lens(28, 3000), 3000, 2000)
        assert spot.width == 3000

    def test_the_cap_is_a_cap(self):
        assert vr180.patch(lens(28, 9000), 9000, 6000, cap=4096).width == 4096

    def test_an_explicit_size_is_taken(self):
        assert vr180.patch(lens(28, 3000), 3000, 2000, size=1024).width == 1024

    def test_both_dimensions_come_out_even(self):
        spot = vr180.patch(lens(35, 999), 999, 667)
        assert spot.width % 2 == 0 and spot.height % 2 == 0

    def test_pixels_per_radian_follows_the_span(self):
        """Half the angle in the same pixels is twice the density, and the
        disparity is measured against it."""
        wide = vr180.patch(lens(14, 1000), 1000, 1000, size=1000)
        narrow = vr180.patch(lens(50, 1000), 1000, 1000, size=1000)
        assert narrow.per_radian > wide.per_radian


class TestCropPaysForItself:
    """The reason cropping exists, stated as a number rather than a hope."""

    @pytest.mark.parametrize("equivalent_mm,gain", [(16, 1.86), (28, 2.75), (49, 4.47)])
    def test_the_picture_gets_the_whole_width_instead_of_its_share(self, equivalent_mm, gain):
        """In the square a lens got only its angular share of the pixels -- 180
        degrees of frame, and the picture covering however much of it the lens
        managed.  Cropped, the whole stored width is picture, and the gain is
        exactly the reciprocal of the share it used to get."""
        src, stored = 1200, 600
        focal = lens(equivalent_mm, src)
        cropped = vr180.patch(focal, src, src, size=stored)
        in_square = stored * vr180.fov(focal, src) / 180.0
        assert cropped.width == stored
        assert cropped.width / in_square == pytest.approx(gain, rel=0.01)

    def test_a_narrow_lens_gains_the_most(self):
        """A 49mm lens fills 5% of a hemisphere against a 16mm's 30%, so it is
        the one the square treated worst and the one cropping rescues hardest."""
        src = 1200
        wide = 180.0 / vr180.fov(lens(16, src), src)
        narrow = 180.0 / vr180.fov(lens(49, src), src)
        assert narrow > wide * 2


class TestProjection:
    def test_the_centre_of_the_patch_is_the_centre_of_the_photo(self):
        # Odd sizes, so that "the centre" is a pixel rather than the crack
        # between two of them.
        spot = vr180.Patch(180.0, 180.0, 33, 33)
        out, mask = vr180.project(dot(65), focal_px=65.0, spot=spot)
        assert bool(mask[16, 16])
        assert divmod(int(torch.argmax(out[0])), 33) == (16, 16)

    @pytest.mark.parametrize("col", [180, 200, 240, 255])
    def test_an_azimuth_lands_where_the_tangent_says(self, col):
        """A ray at theta off-axis meets the sensor at f*tan(theta), which is the
        whole of what "perspective" means and the one thing worth checking."""
        spot, focal, src = vr180.Patch(180.0, 180.0, 360, 360), 500.0, 800
        grid, valid = vr180._grid(180, 1, spot, focal, src, src,
                                  torch.device("cpu"), torch.float32)
        azimuth = math.radians((col + 0.5) / spot.width * spot.span_az - spot.span_az / 2)
        u = ((float(grid[0, 0, col, 0]) + 1.0) * src - 1.0) / 2.0
        assert u == pytest.approx((src - 1) / 2 + focal * math.tan(azimuth), abs=0.01)
        assert bool(valid[0, col])

    def test_a_cropped_patch_reads_the_same_angles_as_a_full_one(self):
        """Cropping must move nothing.  A column at a given bearing has to come
        from the same place in the photograph either way, or the picture ends up
        at the wrong scale in a headset while looking perfectly sharp."""
        focal, src = 500.0, 800
        full = vr180.Patch(180.0, 180.0, 720, 720)
        spot = vr180.patch(focal, src, src, size=720)
        bearing = math.radians(10.0)

        def sampled(p):
            col = int(round((math.degrees(bearing) + p.span_az / 2) / p.span_az * p.width - 0.5))
            grid, _ = vr180._grid(p.height // 2, 1, p, focal, src, src,
                                  torch.device("cpu"), torch.float32)
            return ((float(grid[0, 0, col, 0]) + 1.0) * src - 1.0) / 2.0

        assert sampled(spot) == pytest.approx(sampled(full), abs=1.5)

    def test_nothing_behind_the_camera_is_ever_valid(self):
        spot = vr180.Patch(180.0, 180.0, 64, 64)
        _, mask = vr180.project(torch.ones(1, 64, 64), focal_px=64.0, spot=spot)
        assert not bool(mask[:, 0].any()), "the left pole looks backwards"
        assert not bool(mask[:, -1].any()), "and so does the right one"

    @pytest.mark.parametrize("focal", [128.0, 256.0, 512.0])
    def test_coverage_is_the_share_of_the_hemisphere_the_lens_saw(self, focal):
        """The solid angle a rectilinear frame covers is 4*asin(sin(a/2)sin(b/2)),
        and a hemisphere is 2pi of them."""
        spot, src = vr180.Patch(180.0, 180.0, 512, 512), 256
        _, mask = vr180.project(torch.ones(1, src, src), focal, spot)
        a = b = math.radians(vr180.fov(focal, src))
        expected = 4 * math.asin(math.sin(a / 2) * math.sin(b / 2)) / (2 * math.pi)
        assert vr180.coverage(mask, spot) == pytest.approx(expected, rel=0.02)

    def test_cropping_does_not_change_how_much_sphere_is_real(self):
        """Fewer stored pixels, the same picture on the same sphere.  If this
        moves, the crop has quietly changed the field of view."""
        focal, src = 256.0, 256
        full = vr180.Patch(180.0, 180.0, 512, 512)
        spot = vr180.patch(focal, src, src, size=512)
        _, full_mask = vr180.project(torch.ones(1, src, src), focal, full)
        _, crop_mask = vr180.project(torch.ones(1, src, src), focal, spot)
        assert vr180.coverage(crop_mask, spot) == pytest.approx(
            vr180.coverage(full_mask, full), rel=0.03)

    def test_a_cropped_patch_is_mostly_picture(self):
        """The complaint that started this: the square was 97% black."""
        focal, src = 256.0, 256
        spot = vr180.patch(focal, src, src, size=256)
        _, mask = vr180.project(torch.ones(1, src, src), focal, spot)
        assert float(mask.float().mean()) > 0.85


class TestOdsDisparity:
    """dtheta = B*(1/Z - 1/Zc) radians, tapering to nothing at the poles."""

    def spot(self, span=180.0, width=360):
        return vr180.Patch(span, span, width, width)

    def test_matches_the_formula_at_the_equator(self):
        spot, z, focus = self.spot(), 5.0, 3.0
        half = vr180.half_disparity(inv(z), spot, 65.0, focus, elevation=torch.zeros(1))
        radians = float(half) * 2 / spot.per_radian
        assert radians == pytest.approx(0.065 * (1 / z - 1 / focus), abs=1e-6)

    def test_the_angle_is_the_same_whatever_the_patch(self):
        """Pixels change with the crop; the geometry must not."""
        z, focus = 5.0, 3.0
        wide = vr180.half_disparity(inv(z), self.spot(180.0, 360), 65.0, focus,
                                    elevation=torch.zeros(1))
        tight = vr180.half_disparity(inv(z), self.spot(45.0, 360), 65.0, focus,
                                     elevation=torch.zeros(1))
        assert float(wide) * 2 / self.spot(180.0, 360).per_radian == pytest.approx(
            float(tight) * 2 / self.spot(45.0, 360).per_radian, rel=1e-5)

    def test_zero_at_the_screen_plane(self):
        half = vr180.half_disparity(inv(3.0), self.spot(), 65.0, 3.0, elevation=torch.zeros(1))
        assert float(half) == pytest.approx(0.0, abs=1e-9)

    def test_the_poles_have_no_separation_to_give(self):
        """Looking straight up there is no across-the-line-of-sight left to put
        two eyes on, and a projection that asks for parallax there is asking for
        something no pair of eyes could produce."""
        poles = torch.tensor([math.pi / 2, 0.0, -math.pi / 2])
        half = vr180.half_disparity(inv(1.0).expand(3, 1), self.spot(), 65.0, 3.0,
                                    elevation=poles)
        up, level, down = half[:, 0].tolist()
        assert up == pytest.approx(0.0, abs=1e-6)
        assert down == pytest.approx(0.0, abs=1e-6)
        assert abs(level) > 0

    def test_the_taper_is_a_cosine(self):
        at = torch.tensor([0.0, math.pi / 3])  # 60 degrees up, so half the effect
        half = vr180.half_disparity(inv(1.0).expand(2, 1), self.spot(), 65.0, 3.0, elevation=at)
        assert float(half[1, 0]) == pytest.approx(float(half[0, 0]) * 0.5, rel=1e-4)

    def test_the_clamp_is_in_degrees(self):
        spot, limit = self.spot(), 1.2
        half = vr180.half_disparity(inv(0.02), spot, 500.0, 3.0, limit_deg=limit,
                                    elevation=torch.zeros(1))
        assert float(half.abs()) <= math.radians(limit) * spot.per_radian / 2 + 1e-6

    def test_defaults_to_one_elevation_per_row(self):
        spot = vr180.Patch(180.0, 180.0, 8, 8)
        assert vr180.half_disparity(torch.full((8, 8), 0.5), spot, 65.0, 3.0).shape == (8, 8)


class TestRender:
    def spot(self, w=32, h=32, span=180.0):
        return vr180.Patch(span, span * h / w, w, h)

    def test_the_pair_is_the_size_the_patch_asked_for(self):
        rgb = torch.rand(3, 48, 64)
        spot = self.spot(40, 30)
        left, right, mask = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, spot)
        assert left.shape == right.shape == (3, 30, 40)
        assert mask.shape == (30, 40)

    def test_nothing_is_trimmed(self):
        """The flat path trims the sliver only one eye reaches.  Here that sliver
        is angle, and trimming it would put every remaining pixel at the wrong
        bearing while looking perfectly fine."""
        rgb = torch.rand(3, 48, 64)
        left, _, _ = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, self.spot(40, 40))
        assert left.shape[2] == 40

    def test_the_void_stays_dark(self):
        rgb = torch.ones(3, 64, 64)
        spot = self.spot(64, 64)
        left, right, mask = vr180.render(rgb, torch.full((64, 64), 0.5), 64.0, 65.0, 3.0, spot)
        assert float(left[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)
        assert float(right[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)

    def test_a_tight_crop_is_not_darkened_all_round_its_edge(self):
        """The fade belongs at the boundary of the picture, not at the boundary
        of the file.  Cropped to its own content there is no room to spare, and
        fading anyway would vignette every photo."""
        rgb = torch.ones(3, 64, 96)
        spot = vr180.patch(96.0, 96, 64, size=96)
        left, _, _ = vr180.render(rgb, torch.full((64, 96), 1 / 3), 96.0, 65.0, 3.0, spot)
        middle = left[0, spot.height // 2]
        assert float(middle[len(middle) // 2]) == pytest.approx(1.0, abs=1e-3)
        assert float(middle[2]) > 0.9, "the left edge is picture, not fade"

    def test_a_flat_scene_leaves_the_two_eyes_alike(self):
        """Everything at the screen plane has no separation, so the pair matches."""
        rgb = torch.rand(3, 64, 64)
        left, right, _ = vr180.render(rgb, torch.full((64, 64), 1.0 / 3.0), 64.0, 65.0, 3.0,
                                      self.spot(48, 48))
        assert torch.allclose(left, right, atol=1e-5)

    def test_depth_moves_the_eyes_apart(self):
        rgb = torch.rand(3, 64, 64)
        left, right, _ = vr180.render(rgb, torch.full((64, 64), 1 / 0.8), 64.0, 200.0, 3.0,
                                      self.spot(48, 48))
        assert not torch.allclose(left, right, atol=1e-3)


class TestAutoTarget:
    """The separation `auto` asks for has to be an angle, and stay one however
    much of the sphere the file happens to store."""

    def scene(self):
        return 1.0 / torch.linspace(2.0, 40.0, 4096)[None]

    def chosen(self, spot):
        eyes, _ = stereo.auto_geometry(self.scene(), spot.width, spot.per_radian,
                                       vr180.auto_target(spot))
        return eyes

    def test_a_full_frame_asks_for_its_share_of_180(self):
        assert vr180.auto_target(vr180.Patch(180.0, 180.0, 8, 8)) == pytest.approx(
            100.0 * vr180.TARGET_DEG / 180.0)

    def test_a_crop_asks_for_its_share_of_the_crop(self):
        """The share is of the frame, and the frame got smaller."""
        assert vr180.auto_target(vr180.Patch(65.5, 51.5, 8, 8)) == pytest.approx(
            100.0 * vr180.TARGET_DEG / 65.5)

    @pytest.mark.parametrize("span", [180.0, 96.0, 65.5, 40.3])
    def test_the_shared_geometry_hits_that_angle(self, span):
        """Feed `auto_geometry` the patch's pixels-per-radian and it should pick a
        baseline that spreads the scene over `TARGET_DEG` degrees -- whatever the
        patch, because degrees of arc do not care how the file was cut."""
        spot = vr180.Patch(span, span, 360, 360)
        lo, hi = torch.quantile(self.scene().flatten(), torch.tensor([0.02, 0.98]))
        assert (self.chosen(spot) / 1000.0) * float(hi - lo) == pytest.approx(
            math.radians(vr180.TARGET_DEG), rel=1e-3)

    def test_cropping_does_not_change_the_baseline(self):
        """The bug this replaced: pinned at 180, a 65-degree patch came out with
        under half the separation the same scene got in the square."""
        full = self.chosen(vr180.Patch(180.0, 180.0, 1768, 1768))
        crop = self.chosen(vr180.Patch(65.5, 51.5, 642, 505))
        assert crop == pytest.approx(full, rel=0.02)
