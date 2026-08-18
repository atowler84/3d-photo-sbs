"""The video plumbing: probing, geometry, encoder choice, temporal smoothing.

None of this needs the depth model.  It does need ffmpeg, which video needs
anyway.
"""

import subprocess

import pytest
import torch

from stereocraft import video
from stereocraft.pipeline import Settings, VideoSettings


class TestProbe:
    def test_reads_the_shape_of_a_clip(self, clip):
        c = video.probe(clip)
        assert (c.width, c.height) == (320, 240)
        assert c.fps == pytest.approx(30.0)
        assert c.frames == 60 and c.audio == "aac"

    def test_a_silent_clip_has_no_audio(self, silent_clip):
        assert video.probe(silent_clip).audio is None

    def test_rotation_is_reported_the_way_the_decoder_will_hand_it_over(self, clip, tmp_path):
        """Phones record sideways and note the rotation instead of turning the
        pixels; probe has to agree with what ffmpeg actually decodes."""
        turned = tmp_path / "turned.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
                        "-i", str(clip), "-c", "copy", str(turned)], check=True)
        c = video.probe(turned)
        assert (c.width, c.height) == (240, 320)

    def test_something_with_no_picture_in_it_is_refused(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("not a video")
        with pytest.raises(ValueError):
            video.probe(junk)

    def test_a_still_reads_as_a_one_frame_clip(self, photo):
        """ffprobe sees a JPEG as a single-frame mjpeg stream, so probe does not
        refuse it.  Nothing reaches here by that route -- the CLI and the window
        both sort by extension first -- but it is worth knowing which end the
        filtering happens at."""
        assert video.probe(photo).frames <= 1


class TestGeometry:
    def test_half_width_puts_a_clip_out_the_size_it_came_in(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3), VideoSettings())
        assert g.width == 1920 and g.height == 1080

    def test_full_width_keeps_every_native_pixel(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3), VideoSettings(full_width=True))
        assert g.width > 1920

    @pytest.mark.parametrize("w,h", [(1921, 1081), (641, 361), (100, 99)])
    def test_both_dimensions_come_out_even(self, w, h):
        """yuv420p halves both, so an odd one cannot be encoded."""
        g = video.geometry(video.Clip(w, h, 30.0, 10, 1.0), VideoSettings())
        assert g.width % 2 == 0 and g.height % 2 == 0

    def test_the_margin_is_pinned_rather_than_measured(self):
        """Frames trimmed by different amounts come out different sizes, which
        no encoder will take."""
        clip = video.Clip(1920, 1080, 30.0, 100, 3.3)
        settings = VideoSettings()
        from stereocraft import stereo
        assert video.geometry(clip, settings).margin == stereo.max_margin(1920, settings.limit_pct)


class TestVr180Geometry:
    """A clip on a sphere: square per eye, no trim, and sized without the lens."""

    def test_each_eye_is_square_and_the_pair_is_two_to_one(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3),
                           VideoSettings(projection="vr180"))
        assert g.eye == g.height
        assert g.width == 2 * g.height

    def test_nothing_is_trimmed(self):
        """The sliver only one eye reaches is angle here, and cutting it would
        put every remaining pixel at the wrong bearing."""
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3),
                           VideoSettings(projection="vr180"))
        assert g.margin == 0

    def test_an_explicit_size_is_taken(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3),
                           VideoSettings(projection="vr180", vr180_size=1024))
        assert g.eye == 1024

    def test_a_small_clip_is_not_blown_up_to_the_cap(self):
        """The cap is a ceiling, not a target.  Going straight to it would
        upscale a 320-wide clip more than six times over for nothing."""
        g = video.geometry(video.Clip(320, 240, 30.0, 30, 1.0),
                           VideoSettings(projection="vr180"))
        assert g.eye < VideoSettings.vr180_cap
        assert g.eye == pytest.approx(320 * 180 / 65.47, abs=2)

    def test_a_large_one_stops_at_the_cap(self):
        g = video.geometry(video.Clip(3840, 2160, 30.0, 30, 1.0),
                           VideoSettings(projection="vr180"))
        assert g.eye == VideoSettings.vr180_cap

    @pytest.mark.parametrize("w", [321, 641, 1001])
    def test_the_square_comes_out_even(self, w):
        g = video.geometry(video.Clip(w, w, 30.0, 10, 1.0), VideoSettings(projection="vr180"))
        assert g.width % 2 == 0 and g.height % 2 == 0


class TestEncoders:
    def test_prefers_x264_when_it_is_there(self):
        video._ENCODERS_SEEN = None
        name, quality, _ = video.pick_encoder("h264")
        assert name == "libx264" and quality == "-crf"

    def test_falls_past_it_when_it_is_not(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"h264_nvenc", "libopenh264"})
        name, quality, _ = video.pick_encoder("h264")
        assert name == "h264_nvenc" and quality == "-cq"

    def test_falls_all_the_way_to_software(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"libopenh264"})
        assert video.pick_encoder("h264")[0] == "libopenh264"

    def test_says_so_rather_than_writing_nothing(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"mjpeg"})
        with pytest.raises(video.MissingFFmpeg):
            video.pick_encoder("h264")


class TestNoConsole:
    """Every ffmpeg has to be launched with whatever it takes to keep Windows
    from giving it a console window of its own: the window has none, so each
    child would flash a black box up over the screen -- and the encoder's would
    sit there for the length of the clip."""

    def test_the_probe_asks_for_no_window(self, monkeypatch, clip):
        seen = {}
        real = video.subprocess.run

        def watched(args, **kwargs):
            seen.update(kwargs)
            return real(args, **kwargs)

        monkeypatch.setattr(video.subprocess, "run", watched)
        video.probe(clip)
        assert seen.items() >= video.NO_CONSOLE.items()

    def test_and_so_do_the_two_that_do_the_work(self, monkeypatch, clip, tmp_path):
        seen = []
        info = video.probe(clip)  # before Popen is stood on, since it needs it
        settings = video.VideoSettings()
        monkeypatch.setattr(video.subprocess, "Popen", lambda args, **kwargs: seen.append(kwargs))
        video._decoder(clip, info, None, None)
        video._encoder(tmp_path / "out.mp4", clip, info, video.geometry(info, settings),
                       settings, None)
        assert len(seen) == 2
        assert all(kwargs.items() >= video.NO_CONSOLE.items() for kwargs in seen)


class TestTemporalDepth:
    def test_the_first_frame_passes_through(self):
        d = torch.rand(1, 1, 8, 8)
        assert torch.allclose(video.TemporalDepth(0.5)(d), d)

    def test_smoothing_pulls_towards_the_frame_before(self):
        """A modest change, since a large one is a cut and handled differently."""
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        out = smooth(torch.full((1, 1, 8, 8), 1.2))
        assert float(out.mean()) == pytest.approx(1.1, abs=1e-5)
        assert smooth.cuts == 0

    def test_zero_keeps_nothing(self):
        smooth = video.TemporalDepth(0.0)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        assert float(smooth(torch.full((1, 1, 8, 8), 1.2)).mean()) == pytest.approx(1.2)

    def test_it_never_freezes_the_first_frame_forever(self):
        assert video.TemporalDepth(1.0).keep <= 0.95

    def test_a_cut_starts_the_memory_again(self):
        """Averaging across a cut would drag one scene into the next."""
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 0.1))
        after = smooth(torch.full((1, 1, 8, 8), 5.0))
        assert smooth.cuts == 1
        assert float(after.mean()) == pytest.approx(5.0), "the new scene arrives unmixed"

    def test_an_ordinary_change_is_not_a_cut(self):
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        smooth(torch.full((1, 1, 8, 8), 1.05))
        assert smooth.cuts == 0


class TestOutputPath:
    def test_names_the_output_after_the_input(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov").name == "a_sbs.mp4"

    def test_a_folder_gets_the_generated_name(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov", tmp_path).name == "a_sbs.mp4"

    def test_an_explicit_name_is_kept(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov", tmp_path / "b.mkv").name == "b.mkv"


class TestClock:
    @pytest.mark.parametrize("seconds,expected", [(5, "5s"), (65, "1m05s"), (3600, "1h00m")])
    def test_formats_a_wait(self, seconds, expected):
        assert video.clock(seconds) == expected
