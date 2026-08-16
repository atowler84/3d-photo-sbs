"""The geometry, which is the part that has to be exactly right.

Everything else in this app is plumbing around these few lines: if the disparity
field is wrong the picture is wrong, and no amount of correct file handling
saves it.
"""

import math

import pytest
import torch

from stereocraft import stereo


def inv(*metres):
    return torch.tensor([[1.0 / z for z in metres]])


class TestHalfDisparity:
    """d = f*B*(1/Z - 1/Zc), the shifted-sensor arrangement."""

    @pytest.mark.parametrize("z", [0.5, 1.0, 3.0, 10.0, 1e6])
    def test_matches_the_formula(self, z):
        f, b, zc, width = 1000.0, 0.065, 3.0, 2000
        got = float(stereo.half_disparity(inv(z), f, 65.0, zc, limit=1e6, width=width)) * 2
        assert got == pytest.approx(f * b * (1 / z - 1 / zc), abs=1e-4)

    def test_zero_at_the_screen_plane(self):
        assert float(stereo.half_disparity(inv(3.0), 1000.0, 65.0, 3.0)) == pytest.approx(0, abs=1e-6)

    def test_near_comes_forward_and_far_recedes(self):
        half = stereo.half_disparity(inv(1.0, 3.0, 50.0), 1000.0, 65.0, 3.0)
        near, plane, far = half[0].tolist()
        assert near > 0 and far < 0 and plane == pytest.approx(0, abs=1e-6)

    def test_distant_things_converge_rather_than_stretch(self):
        """The whole point of metric depth: separation approaches a finite limit
        as things recede, and never passes it, where a normalised depth map would
        have stretched the far end to fill whatever range it was given."""
        f, b, zc = 1000.0, 0.065, 3.0
        limit = -f * b / zc
        got = [float(stereo.half_disparity(inv(z), f, 65.0, zc)) * 2 for z in (1e2, 1e3, 1e6)]
        assert all(g > limit for g in got), "nothing may exceed the limit"
        assert got == sorted(got, reverse=True), "further away must mean closer to it"
        # float32 lands a fraction the far side of the limit at absurd distances,
        # which is rounding rather than geometry, so the last check allows for it.
        assert float(stereo.half_disparity(inv(1e9), f, 65.0, zc)) * 2 == pytest.approx(limit, abs=1e-4)

    def test_a_wider_baseline_scales_the_whole_field(self):
        one = stereo.half_disparity(inv(1.0, 5.0, 20.0), 1000.0, 65.0, 3.0)
        two = stereo.half_disparity(inv(1.0, 5.0, 20.0), 1000.0, 130.0, 3.0)
        assert torch.allclose(two, one * 2, atol=1e-5)

    def test_the_clamp_bounds_something_very_close(self):
        width, limit = 2000, 3.0
        half = stereo.half_disparity(inv(0.05), 1000.0, 65.0, 3.0, limit=limit, width=width)
        assert float(half.abs().max()) <= limit / 100 * width / 2 + 1e-6

    def test_no_clamp_without_a_width(self):
        assert float(stereo.half_disparity(inv(0.05), 1000.0, 65.0, 3.0, limit=3.0)) > 100


class TestMargin:
    @pytest.mark.parametrize("width,limit", [(1000, 3.0), (1920, 1.3), (640, 4.0), (100, 0.5)])
    def test_covers_whatever_the_clamp_allows(self, width, limit):
        """A clip trims by this, so it has to be at least what any frame needs."""
        margin = stereo.max_margin(width, limit)
        worst = limit / 100 * width / 2
        assert margin >= math.floor(worst)
        assert margin <= (width - 1) // 2

    def test_never_eats_the_whole_frame(self):
        assert stereo.max_margin(10, 500.0) <= 4


class TestNormalise:
    def test_percentiles_and_apply_range_round_trip(self):
        x = torch.linspace(0, 1, 1000).reshape(1, 1, 10, 100)
        lo, hi = stereo.percentiles(x, 0.0, 100.0)
        out = stereo.apply_range(x, lo, hi)
        assert float(out.min()) == pytest.approx(0, abs=1e-5)
        assert float(out.max()) == pytest.approx(1, abs=1e-5)

    def test_a_flat_map_does_not_divide_by_zero(self):
        flat = torch.full((1, 1, 4, 4), 0.5)
        assert torch.all(stereo.apply_range(flat, *stereo.percentiles(flat)) == 0)

    def test_outliers_cannot_squash_the_range(self):
        x = torch.cat([torch.full((998,), 0.5), torch.tensor([-99.0, 99.0])]).reshape(1, 1, 10, 100)
        out = stereo.normalize(x)
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


class TestCompose:
    def test_left_then_right_by_default(self):
        left, right = torch.zeros(3, 4, 5), torch.ones(3, 4, 5)
        out = stereo.compose(left, right)
        assert out.shape == (3, 4, 10)
        assert float(out[:, :, :5].max()) == 0 and float(out[:, :, 5:].min()) == 1

    def test_cross_eyed_swaps_them(self):
        left, right = torch.zeros(3, 4, 5), torch.ones(3, 4, 5)
        out = stereo.compose(left, right, cross_eyed=True)
        assert float(out[:, :, :5].min()) == 1 and float(out[:, :, 5:].max()) == 0


class TestMakePair:
    def test_trims_both_ends_by_the_margin(self):
        image = torch.rand(3, 8, 100)
        half = torch.zeros(8, 100)
        left, right = stereo.make_pair(image, half, margin=7)
        assert left.shape[2] == right.shape[2] == 100 - 14

    def test_zero_disparity_leaves_the_picture_alone(self):
        image = torch.rand(3, 8, 40)
        left, right = stereo.make_pair(image, torch.zeros(8, 40), margin=0)
        assert torch.allclose(left, image, atol=1e-5) and torch.allclose(right, image, atol=1e-5)

    def test_the_eyes_differ_once_there_is_depth(self):
        image = torch.rand(3, 8, 60)
        half = torch.linspace(-4, 4, 60).expand(8, 60).contiguous()
        left, right = stereo.make_pair(image, half, margin=6)
        assert not torch.allclose(left, right, atol=1e-3)
