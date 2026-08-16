"""End to end, with the depth model loaded.  Slow, and the part that matters."""

import numpy as np
import pytest
from PIL import Image

from stereocraft.pipeline import Settings, save_depth_map
import torch

pytestmark = pytest.mark.slow


class TestDepthMap:
    def test_uses_the_whole_range_so_it_can_be_seen(self, tmp_path):
        """It once stored centimetres, which is correct and unreadable: a scene
        two to twenty-five metres away came out under 4% of full scale, which is
        a black rectangle."""
        inverse = torch.linspace(1 / 25.0, 1 / 2.0, 64).reshape(8, 8)
        path = save_depth_map(inverse, tmp_path / "d.png")
        d = np.array(Image.open(path))
        assert d.max() > 60000, "the nearest thing should be near white"
        assert d.min() < 5000, "the furthest should be near black"

    def test_the_metres_survive_in_the_metadata(self, tmp_path):
        inverse = torch.linspace(1 / 25.0, 1 / 2.0, 64).reshape(8, 8)
        image = Image.open(save_depth_map(inverse, tmp_path / "d.png"))
        near = float(image.text["stereocraft:near_m"])
        far = float(image.text["stereocraft:far_m"])
        assert near == pytest.approx(2.0, rel=1e-3) and far == pytest.approx(25.0, rel=1e-3)

        d = np.array(image).astype(float)
        recovered = far - (d / 65535) * (far - near)
        assert recovered.min() == pytest.approx(2.0, rel=1e-2)
        assert recovered.max() == pytest.approx(25.0, rel=1e-2)

    def test_near_is_brighter_than_far(self, tmp_path):
        inverse = torch.tensor([[1 / 20.0, 1 / 2.0]])
        d = np.array(Image.open(save_depth_map(inverse, tmp_path / "d.png")))
        assert d[0, 1] > d[0, 0]

    def test_a_scene_at_one_distance_does_not_divide_by_zero(self, tmp_path):
        flat = torch.full((4, 4), 0.25)
        d = np.array(Image.open(save_depth_map(flat, tmp_path / "d.png")))
        assert d.shape == (4, 4)


class TestPhoto:
    def test_converts_and_the_eyes_differ(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        out = np.asarray(Image.open(info["output"])).astype(int)
        w = out.shape[1] // 2
        left, right = out[:, :w], out[:, w:]
        assert not np.array_equal(left, right), "a stereo pair whose eyes match is not one"
        assert left.std() > 5 and right.std() > 5, "neither eye may be blank"

    def test_reports_the_geometry_it_chose(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        assert info["eyes_mm"] > 0 and info["focus_m"] > 0

    def test_the_pair_is_twice_a_frame_less_the_trim(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        width, _ = info["output_size"]
        assert width % 2 == 0 and width <= 2 * 320


class TestVideo:
    def test_every_frame_survives_with_audio(self, converter, clip, tmp_path):
        """-shortest once cost three frames off the end of a ninety-frame clip,
        and the checks at the time counted frame sizes rather than frames."""
        from stereocraft import video
        from stereocraft.pipeline import VideoSettings
        from conftest import audio_codec, frame_count, frame_sizes

        converter.settings = VideoSettings()
        info = video.convert_video(clip, tmp_path / "out.mp4", converter)
        assert info["frames"] == 60
        assert frame_count(info["output"]) == 60
        assert len(frame_sizes(info["output"])) == 1, "every frame must be the same size"
        assert audio_codec(info["output"]) is not None

    def test_stopping_leaves_nothing_behind(self, converter, clip, tmp_path):
        from stereocraft import video
        from stereocraft.pipeline import VideoSettings

        converter.settings = VideoSettings()
        out = tmp_path / "stopped.mp4"
        result = video.convert_video(clip, out, converter,
                                     on_progress=lambda done, total, secs: done < 5)
        assert result is None and not out.exists()
