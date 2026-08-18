"""Argument handling, none of which needs a depth model."""

import pytest

from stereocraft import cli
from stereocraft.pipeline import SBS_TAGS, Settings, VideoSettings, output_path, tag


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


class TestNaming:
    """A player has nothing but the file name to go on -- no JPEG and no mp4 the
    app writes says how it is meant to be looked at -- so the name is the
    setting, and getting it wrong shows the wrong thing rather than nothing."""

    @pytest.mark.parametrize("projection,cross,expected", [
        ("flat", False, "_sbs"),
        ("flat", True, "_sbs_cross"),
        ("vr180", False, "_180_sbs"),
        ("vr180", True, "_180_sbs_cross"),
    ])
    def test_every_combination_gets_its_own_name(self, projection, cross, expected):
        assert tag(Settings(projection=projection, cross_eyed=cross)) == expected

    def test_vr180_carries_both_tokens_a_player_reads(self):
        """180 sets the projection and sbs the layout, and players key on the two
        of them separately -- which is why it is not just "_vr180"."""
        name = tag(Settings(projection="vr180"))
        assert "180" in name and "sbs" in name

    def test_the_flat_name_has_not_moved(self):
        """Everything already converted is called this, and renaming it would
        orphan a library to no purpose."""
        assert tag(Settings()) == "_sbs"

    def test_the_photo_path_uses_it(self, tmp_path):
        out = output_path(tmp_path / "a.jpg", None, "auto", tag(Settings(projection="vr180")))
        assert out.name == "a_180_sbs.jpg"

    def test_the_video_path_uses_it(self, tmp_path):
        from stereocraft import video
        out = video.output_path(tmp_path / "a.mov", None,
                                tag(VideoSettings(projection="vr180", cross_eyed=True)))
        assert out.name == "a_180_sbs_cross.mp4"


class TestCollect:
    def test_finds_photos_and_clips_and_skips_its_own_output(self, tmp_path):
        for name in ("a.jpg", "b.mp4", "c_sbs.jpg", "d_depth.png", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        found = {p.name for p in cli.collect([str(tmp_path)])}
        assert found == {"a.jpg", "b.mp4"}

    @pytest.mark.parametrize("name", SBS_TAGS)
    def test_skips_every_name_it_can_write(self, tmp_path, name):
        (tmp_path / f"a{name}.jpg").write_bytes(b"x")
        (tmp_path / f"b{name}.mp4").write_bytes(b"x")
        assert cli.collect([str(tmp_path)]) == []

    @pytest.mark.parametrize("projection", ["flat", "vr180"])
    @pytest.mark.parametrize("cross", [False, True])
    def test_what_it_writes_is_what_it_then_ignores(self, tmp_path, projection, cross):
        """The round trip, which is the whole point of the list: run it over a
        folder twice and the second pass must find nothing it made on the first.
        A name that gets written but not skipped converts the conversions."""
        settings = Settings(projection=projection, cross_eyed=cross)
        written = output_path(tmp_path / "a.jpg", None, "auto", tag(settings))
        written.write_bytes(b"x")
        assert cli.collect([str(tmp_path)]) == [], f"{written.name} came back round"

    def test_says_so_when_something_is_missing(self, capsys):
        assert cli.collect(["definitely-not-here-*.jpg"]) == []
        assert "not found" in capsys.readouterr().err


class TestClock:
    @pytest.mark.parametrize("seconds,expected", [(0, "0s"), (45, "45s"), (90, "1m30s"), (3700, "1h01m")])
    def test_reads_the_way_someone_waiting_would_say_it(self, seconds, expected):
        assert cli.clock(seconds) == expected
