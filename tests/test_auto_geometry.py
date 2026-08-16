"""Choosing the eye separation and screen plane to suit a scene.

Literal 65mm eyes were measurably the wrong default -- 16% of frame width on a
close-up, 0.3% on a telephoto shot -- so this picks per scene.  What matters is
that it lands near the target whatever the scene, and that the shape of the
disparity is left alone while it does.
"""

import pytest
import torch

from stereocraft import stereo


def scene(near_m, far_m, size=(64, 64)):
    """Inverse depth for a scene running from `near_m` to `far_m`."""
    return torch.linspace(1.0 / far_m, 1.0 / near_m, size[0] * size[1]).reshape(1, 1, *size)


@pytest.mark.parametrize("near,far", [(0.2, 1.0), (0.5, 3.0), (1.9, 25.0), (19.0, 22.0), (2.0, 500.0)])
def test_lands_near_the_target_whatever_the_distance(near, far):
    width, target = 2000, 2.0
    inverse = scene(near, far)
    eyes, focus = stereo.auto_geometry(inverse, width, 1000.0, target=target)
    half = stereo.half_disparity(inverse, 1000.0, eyes, focus, limit=100.0, width=width)
    spread = 100 * float(half.max() - half.min()) * 2 / width
    assert spread == pytest.approx(target, rel=0.25), f"{near}-{far}m gave {spread:.2f}%"


def test_a_close_up_wants_less_than_human_and_a_distant_scene_more():
    close, _ = stereo.auto_geometry(scene(0.2, 1.0), 2000, 1000.0)
    far, _ = stereo.auto_geometry(scene(19.0, 22.0), 2000, 1000.0)
    assert close < 65.0 < far


def test_the_shape_survives_being_rescaled():
    """Only the amplitude is chosen; 1/Z is what makes it realistic."""
    inverse = scene(1.0, 30.0)
    eyes, focus = stereo.auto_geometry(inverse, 2000, 1000.0)
    auto = stereo.half_disparity(inverse, 1000.0, eyes, focus, limit=100.0, width=2000)
    fixed = stereo.half_disparity(inverse, 1000.0, 65.0, focus, limit=100.0, width=2000)
    a, b = auto.flatten(), fixed.flatten()
    a, b = a - a.mean(), b - b.mean()
    assert float((a * b).sum() / (a.norm() * b.norm())) == pytest.approx(1.0, abs=1e-4)


def test_a_scene_at_one_distance_asks_for_nothing():
    eyes, focus = stereo.auto_geometry(torch.full((1, 1, 8, 8), 0.5), 2000, 1000.0)
    assert eyes == 0.0 and focus > 0


def test_the_screen_plane_lands_inside_the_scene():
    for near, far in ((0.3, 2.0), (2.0, 40.0)):
        _, focus = stereo.auto_geometry(scene(near, far), 2000, 1000.0)
        assert near <= focus <= far


def test_most_of_the_scene_ends_up_behind_the_window():
    inverse = scene(0.5, 20.0)
    eyes, focus = stereo.auto_geometry(inverse, 2000, 1000.0)
    half = stereo.half_disparity(inverse, 1000.0, eyes, focus, limit=100.0, width=2000)
    assert float((half < 0).float().mean()) > 0.7
