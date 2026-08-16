"""Argument handling, none of which needs a depth model."""

import pytest

from stereocraft import cli
from stereocraft.pipeline import Settings, VideoSettings


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


class TestRetiredFlags:
    """--disparity and --convergence described something the renderer no longer
    computes.  Translating them would quietly produce a different picture than
    the one asked for, so they stop instead."""

    @pytest.mark.parametrize("flag,replacement", [("--disparity", "--eyes"), ("--convergence", "--focus")])
    def test_say_what_to_use_instead(self, flag, replacement):
        message = cli.retired(parse(flag, "2", "x.jpg"))
        assert message and replacement in message

    def test_silent_when_neither_is_given(self):
        assert cli.retired(parse("x.jpg")) is None

    def test_main_refuses_rather_than_guessing(self, capsys):
        assert cli.main(["--disparity", "2", "x.jpg"]) == 2
        assert "--eyes" in capsys.readouterr().err


class TestNumber:
    @pytest.mark.parametrize("given", [None, "auto", "AUTO"])
    def test_auto_stays_auto(self, given):
        assert cli._number(given) == "auto"

    def test_a_measurement_becomes_a_number(self):
        assert cli._number("65") == 65.0


class TestSettingsFor:
    def test_a_photo_gets_the_photo_defaults(self):
        s = cli.settings_for(parse("x.jpg"), video=False)
        assert isinstance(s, Settings) and not isinstance(s, VideoSettings)
        assert s.eyes_mm == "auto" and s.target_pct == Settings.target_pct

    def test_a_clip_gets_the_gentler_ones(self):
        s = cli.settings_for(parse("x.mp4"), video=True)
        assert isinstance(s, VideoSettings)
        assert s.target_pct == VideoSettings.target_pct < Settings.target_pct

    def test_an_explicit_measurement_wins(self):
        s = cli.settings_for(parse("--eyes", "70", "--focus", "2.5", "x.jpg"), video=False)
        assert s.eyes_mm == 70.0 and s.focus_m == 2.5

    def test_video_flags_reach_the_settings(self):
        s = cli.settings_for(parse("--codec", "hevc", "--crf", "22", "--temporal", "0.8",
                                   "--full", "--no-audio", "x.mp4"), video=True)
        assert (s.codec, s.crf, s.temporal, s.full_width, s.audio) == ("hevc", 22, 0.8, True, False)


class TestCollect:
    def test_finds_photos_and_clips_and_skips_its_own_output(self, tmp_path):
        for name in ("a.jpg", "b.mp4", "c_sbs.jpg", "d_depth.png", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        found = {p.name for p in cli.collect([str(tmp_path)])}
        assert found == {"a.jpg", "b.mp4"}

    def test_says_so_when_something_is_missing(self, capsys):
        assert cli.collect(["definitely-not-here-*.jpg"]) == []
        assert "not found" in capsys.readouterr().err


class TestClock:
    @pytest.mark.parametrize("seconds,expected", [(0, "0s"), (45, "45s"), (90, "1m30s"), (3700, "1h01m")])
    def test_reads_the_way_someone_waiting_would_say_it(self, seconds, expected):
        assert cli.clock(seconds) == expected
