"""Pricing a conversion before it happens.

The point of this module is that a photo too big for the machine is turned away
before it is decoded, rather than by the OOM killer taking the process with it.
So the arithmetic has to be right, and it has to be conservative.
"""

import pytest
import torch

from stereocraft import budget


class FakeEstimator:
    """A depth model's shape without its weights, which is all budget needs."""

    def __init__(self, name="da3", device="cpu"):
        self.name = name
        self.device = torch.device(device)

    def working_size(self, height, width, size="auto"):
        from stereocraft.depth import DA3_MAX_RES, DA3_PATCH, MAX_SIZE, MIN_SIZE, PATCH
        if self.name == "da3":
            longest = max(height, width)
            target = min(DA3_MAX_RES, longest) if size in (None, 0, "auto") else int(size)
            scale, patch = target / longest, DA3_PATCH
        else:
            short = min(height, width)
            scale = (min(MAX_SIZE, max(MIN_SIZE, short)) if size in (None, 0, "auto") else int(size)) / short
            patch = PATCH
        return (max(patch, round(height * scale / patch) * patch),
                max(patch, round(width * scale / patch) * patch))


def test_da3_is_priced_as_a_fixed_cost_plus_pixels():
    """It carries its weights in fp32, so the floor dominates a small input --
    a purely per-pixel model was out by a factor of six at the small end."""
    small = budget.depth_cost(FakeEstimator(), 504, 378, 504)
    big = budget.depth_cost(FakeEstimator(), 2048, 1536, 2048)
    assert small > budget.DA3_FIXED
    assert big > small
    assert small / budget.DA3_FIXED < 1.5, "the fixed cost should dominate a small input"


def test_da2_is_priced_per_pixel_by_model_size():
    args = (1024, 768, 518)
    costs = [budget.depth_cost(FakeEstimator(n), *args) for n in ("da2-small", "da2-base", "da2-large")]
    assert costs == sorted(costs), "a bigger model cannot cost less"


def test_a_conversion_is_priced_at_its_worst_moment():
    """The stages run one after another, so the larger of the two has to fit."""
    est = FakeEstimator()
    w, h = 8000, 6000
    assert budget.needs(est, w, h) == max(budget.depth_cost(est, w, h), w * h * budget.BYTES_PER_PIXEL)


def test_an_unfamiliar_machine_is_given_the_benefit_of_the_doubt(monkeypatch):
    monkeypatch.setattr(budget, "free_bytes", lambda device: None)
    assert budget.fits(FakeEstimator(), 100000, 100000) is True
    assert budget.plan(FakeEstimator(), 100000, 100000) is None


def test_a_photo_that_fits_is_not_offered_a_resize(monkeypatch):
    monkeypatch.setattr(budget, "free_bytes", lambda device: 64 << 30)
    est = FakeEstimator()
    assert budget.fits(est, 4000, 3000)
    assert budget.plan(est, 4000, 3000) is None


def test_a_photo_that_does_not_fit_is_offered_something_smaller(monkeypatch):
    """Room for the network but not for 300 megapixels of frame, which is the
    case a resize actually rescues."""
    free = 12 << 30
    monkeypatch.setattr(budget, "free_bytes", lambda device: free)
    est = FakeEstimator()
    target = budget.plan(est, 20000, 15000)
    assert target is not None and target[0] < 20000
    assert budget.needs(est, *target) <= free * budget.HEADROOM * 1.05


def test_nothing_is_offered_when_the_network_alone_will_not_fit(monkeypatch):
    """Resizing cannot help below the network's own floor, and saying so is the
    difference between advice and 'free some memory'."""
    monkeypatch.setattr(budget, "free_bytes", lambda device: 1 << 30)
    assert budget.plan(FakeEstimator(), 4000, 3000) is None


def test_a_lighter_model_is_suggested_when_one_would_fit(monkeypatch):
    monkeypatch.setattr(budget, "free_bytes", lambda device: 3 << 30)
    assert budget.smaller_model(FakeEstimator("da3"), 4000, 3000) in {"da2-large", "da2-base", "da2-small"}


def test_nothing_is_suggested_when_even_the_smallest_will_not_fit(monkeypatch):
    """Correctly refusing to offer the impossible, rather than naming a model
    that would fail the moment it was tried."""
    monkeypatch.setattr(budget, "free_bytes", lambda device: 2 << 30)
    assert budget.smaller_model(FakeEstimator("da3"), 4000, 3000) is None


def test_nothing_lighter_than_the_smallest(monkeypatch):
    monkeypatch.setattr(budget, "free_bytes", lambda device: 1 << 20)
    assert budget.smaller_model(FakeEstimator("da2-small"), 4000, 3000) is None
