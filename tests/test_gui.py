"""The window's wiring: what the controls are, and what they reach.

None of this needs the depth model -- a conversion is never started -- but it
does need a display, so the whole file steps aside where there is not one.
"""

import threading
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter", reason="this Python was built without Tkinter")
from tkinter import filedialog  # noqa: E402

from stereocraft import gui  # noqa: E402


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError as error:  # a headless machine, or no DISPLAY
        pytest.skip(f"no display: {error}")
    root.withdraw()
    try:
        yield gui.App(root)
    finally:
        root.destroy()


def queued(app, monkeypatch, *names):
    """Put files in the queue the way the user does, dialog and all."""
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kwargs: names)
    app.add_files()


def click(ctrl=False, shift=False):
    """The bits Tk puts in event.state for the modifier keys."""
    return SimpleNamespace(state=(0x0004 if ctrl else 0) | (0x0001 if shift else 0))


def started(app, monkeypatch):
    """The two settings objects a run would have been given.

    Convert hands them to a worker thread, so the thread is what is intercepted
    -- everything upstream of it is the code under test.
    """
    caught = []

    class Fake:
        def __init__(self, target=None, args=(), **kwargs):
            caught.append(args[2])

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", Fake)
    app.start()
    return caught[0]


class TestSettingsReachTheConversion:
    """Every control on the tabs, followed through to the settings a run gets.

    A control that is on the window but wired to nothing looks exactly like one
    that works, right up until the file comes out wrong.
    """

    def test_the_photo_tab(self, app, monkeypatch):
        queued(app, monkeypatch, "holiday.jpg")
        app.photo_depth.set(2.4)
        app.quality.set(88)
        app.fmt.set("png")
        app.max_size.set("Up to 8192 px")
        app.save_depth.set(True)
        photo, _ = started(app, monkeypatch)
        assert photo.target_pct == 2.4
        assert photo.quality == 88 and photo.fmt == "png"
        assert photo.max_size == 8192
        assert photo.save_depth

    def test_the_video_tab(self, app, monkeypatch):
        queued(app, monkeypatch, "clip.mp4")
        app.target.set(1.1)
        app.temporal.set(0.7)
        app.crf.set(21)
        app.codec.set("hevc")
        app.audio.set(False)
        app.full_width.set(True)
        _, video = started(app, monkeypatch)
        assert video.target_pct == 1.1 and video.temporal == 0.7
        assert video.crf == 21 and video.codec == "hevc"
        assert not video.audio and video.full_width

    def test_the_general_tab_reaches_both(self, app, monkeypatch):
        queued(app, monkeypatch, "holiday.jpg")
        app.automatic.set(False)
        app.eyes.set(50.0)
        app.focus.set(1.0 / 4.0)
        app.cross.set(True)
        photo, video = started(app, monkeypatch)
        for settings in (photo, video):
            assert settings.eyes_mm == 50.0
            assert settings.focus_m == pytest.approx(4.0, abs=0.01)
            assert settings.cross_eyed

    def test_matching_the_scene_is_what_auto_means(self, app, monkeypatch):
        queued(app, monkeypatch, "holiday.jpg")
        photo, video = started(app, monkeypatch)
        assert photo.eyes_mm == "auto" and video.focus_m == "auto"


class TestWhatIsOffered:
    """A slider that does nothing where it stands is greyed out, and -- the half
    that once went wrong -- one that does something is not."""

    @staticmethod
    def disabled(widget):
        return "disabled" in widget.state()

    def test_the_eyes_are_the_users_only_when_the_scene_is_not_setting_them(self, app):
        app.automatic.set(True)
        app._refresh_controls()
        assert all(self.disabled(slider) for slider in app._manual)
        app.automatic.set(False)
        app._refresh_controls()
        assert not any(self.disabled(slider) for slider in app._manual)

    def test_depth_is_the_other_way_round(self, app):
        """It is what matching the scene aims for, so it is live exactly when
        the eyes and the focus are not.  These were disabled alongside them."""
        app.automatic.set(True)
        app._refresh_controls()
        assert not any(self.disabled(slider) for slider in app._auto_only)
        app.automatic.set(False)
        app._refresh_controls()
        assert all(self.disabled(slider) for slider in app._auto_only)

    def test_jpeg_quality_only_where_there_is_jpeg(self, app):
        app.fmt.set("png")
        app._refresh_controls()
        assert self.disabled(app.quality_scale)
        app.fmt.set("jpg")
        app._refresh_controls()
        assert not self.disabled(app.quality_scale)


class TestProgress:
    """Every bar has a number beside it, and both come from the same place."""

    def test_a_clips_frames_fill_its_own_row(self, app, monkeypatch):
        queued(app, monkeypatch, "clip.mp4")
        app.events.put(("frame", (0, 45, 300, "2m10s")))
        app._drain()
        assert app.rows[0].percent.cget("text") == "15%"
        assert app.rows[0].bar.cget("value") == 150  # of a maximum of 1000
        assert "frame 45/300" in app.rows[0].detail.cget("text")

    def test_a_photo_moves_rather_than_claiming_a_figure(self, app, monkeypatch):
        """It is over in under a second and has no frames to count, so any
        percentage would be invented."""
        queued(app, monkeypatch, "holiday.jpg")
        app.events.put(("working", 0))
        app._drain()
        assert str(app.rows[0].bar.cget("mode")) == "indeterminate"
        assert app.rows[0].percent.cget("text") == "\u2026"

    def test_a_finished_file_is_full_and_says_so(self, app, monkeypatch, tmp_path):
        from PIL import Image

        out = tmp_path / "holiday_sbs.png"
        Image.new("RGB", (80, 30), "navy").save(out)
        queued(app, monkeypatch, "holiday.jpg")
        app.events.put(("done", (0, {"output": out, "output_size": (80, 30),
                                     "resized_from": None, "seconds": 0.8})))
        app._drain()
        assert app.rows[0].percent.cget("text") == "100%"
        assert app.rows[0].bar.cget("value") == 1000
        assert str(app.rows[0].bar.cget("mode")) == "determinate"

    def test_one_that_went_wrong_claims_nothing(self, app, monkeypatch):
        queued(app, monkeypatch, "clip.mp4")
        app.events.put(("error", (0, "clip.mp4: ffmpeg fell over")))
        app._drain()
        assert app.rows[0].percent.cget("text") == "\u2013"

    def test_the_main_bar_counts_the_queue_in_percent(self, app, monkeypatch):
        queued(app, monkeypatch, "one.jpg", "two.jpg", "three.jpg", "four.jpg")
        app._set_progress(0, maximum=len(app.files))
        assert app.progress_percent.cget("text") == "0%"
        app.finished = 1
        app._set_progress(app.finished)
        assert app.progress_percent.cget("text") == "25%"
        # A clip advances it a fraction of a file at a time rather than in jumps.
        app.events.put(("frame", (1, 150, 300, "")))
        app._drain()
        assert app.progress_percent.cget("text") == "38%"

    def test_it_never_divides_by_an_empty_queue(self, app):
        app._set_progress(0, maximum=0)
        assert app.progress_percent.cget("text") == "0%"


class TestReset:
    """One button per tab, putting back that tab and nothing else.

    Nothing else matters: a Reset that reached across the window would undo work
    on a tab the user cannot currently see.
    """

    @staticmethod
    def button(app, tab):
        return app._resets[tab][0]

    def test_the_photo_tab_goes_back_and_the_others_stay(self, app):
        app.quality.set(72)
        app.fmt.set("png")
        app.max_size.set("Up to 4096 px")
        app.save_depth.set(True)
        app.photo_depth.set(3.2)
        app.crf.set(25)  # a video setting, which this Reset has no business in
        app.cross.set(True)
        app.reset_photo()
        assert app.quality.get() == 95 and app.fmt.get() == "auto"
        assert app.max_size.get() == "Native size" and not app.save_depth.get()
        assert app.photo_depth.get() == 2.0
        assert app.crf.get() == 25 and app.cross.get()

    def test_the_video_tab_goes_back_and_the_others_stay(self, app):
        app.target.set(2.6)
        app.temporal.set(0.0)
        app.crf.set(25)
        app.codec.set("hevc")
        app.audio.set(False)
        app.full_width.set(True)
        app.quality.set(72)
        app.reset_video()
        assert app.target.get() == 1.3 and app.temporal.get() == 0.5
        assert app.crf.get() == 18 and app.codec.get() == "h264"
        assert app.audio.get() and not app.full_width.get()
        assert app.quality.get() == 72

    def test_the_general_tab_goes_back_and_the_others_stay(self, app):
        app.automatic.set(False)
        app.eyes.set(30.0)
        app.focus.set(1.0 / 12.0)
        app.cross.set(True)
        app.save_depth.set(True)
        app.reset_general()
        assert app.automatic.get() and not app.cross.get()
        assert app.eyes.get() == 65.0 and app.focus.get() == pytest.approx(1 / 3.0)
        assert app.save_depth.get()

    def test_what_general_goes_back_to_follows_the_queue(self, app, monkeypatch):
        """A clip is recommended gentler eyes than a still, so that is what its
        default is while a clip is what is queued."""
        queued(app, monkeypatch, "clip.mp4")
        app.eyes.set(80.0)
        app.reset_general()
        assert app.eyes.get() == 45.0

    @pytest.mark.parametrize("tab,change", [
        (0, lambda app: app.cross.set(True)),
        (1, lambda app: app.quality.set(72)),
        (2, lambda app: app.audio.set(False)),
    ])
    def test_grey_until_there_is_something_to_undo(self, app, tab, change):
        assert "disabled" in self.button(app, tab).state()
        change(app)
        assert "disabled" not in self.button(app, tab).state()

    def test_and_grey_again_afterwards(self, app):
        app.temporal.set(0.1)
        assert "disabled" not in self.button(app, 2).state()
        app.reset_video()
        assert "disabled" in self.button(app, 2).state()


class TestTheResultArea:
    """It is a dark panel, and a dark panel with nothing in it and nothing to
    say about that is indistinguishable from a window that has broken."""

    def test_it_says_what_will_arrive_in_it(self, app):
        assert not app.canvas.cget("image")
        assert "appears here" in app.canvas.cget("text")

    def test_a_picture_replaces_the_words(self, app):
        image = tk.PhotoImage(master=app.root, width=8, height=4)
        app._show_result(image, "holiday_sbs.jpg  -  2000x750")
        assert app.canvas.cget("image") and not app.canvas.cget("text")
        assert app.caption.cget("text").startswith("holiday_sbs.jpg")

    def test_and_the_words_come_back_when_the_queue_is_emptied(self, app, monkeypatch):
        app._show_result(tk.PhotoImage(master=app.root, width=8, height=4), "holiday_sbs.jpg")
        queued(app, monkeypatch, "holiday.jpg")
        app.clear()
        assert not app.canvas.cget("image")
        assert "appears here" in app.canvas.cget("text")


    def test_the_caption_says_how_to_look_at_it(self, app, monkeypatch, tmp_path):
        """Which eye is which is the one thing a side-by-side file cannot tell
        you afterwards, and it is decided by a checkbox halfway up the window."""
        from PIL import Image

        out = tmp_path / "holiday_sbs.png"
        Image.new("RGB", (80, 30), "navy").save(out)
        queued(app, monkeypatch, "holiday.jpg")
        app.cross_used = True
        app.events.put(("done", (0, {"output": out, "output_size": (80, 30),
                                     "resized_from": None, "seconds": 0.8})))
        app._drain()
        assert "cross-eyed order" in app.caption.cget("text")


class TestTheQueue:
    """A row of widgets per file rather than a line of text, so each one can
    show a bar of its own -- which is what a clip, alone in taking minutes,
    actually needs."""

    def test_a_row_per_file_saying_which_file(self, app, monkeypatch):
        queued(app, monkeypatch, "holiday.jpg", "clip.mp4")
        assert len(app.rows) == 2
        assert "holiday.jpg" in app.rows[0].name.cget("text")
        assert "clip.mp4" in app.rows[1].name.cget("text")

    def test_a_long_name_keeps_its_ends(self, app, monkeypatch):
        """The extension says what the file is, and a batch off a camera differs
        only in the digits just before it -- so the middle is what goes."""
        name = "a_very_long_holiday_video_from_the_camera_20260816_143200.mp4"
        queued(app, monkeypatch, name)
        shown = app.rows[0].name.cget("text")
        assert len(shown) < len(name)
        assert shown.startswith("   a_very_long") and shown.endswith("143200.mp4")

    def test_an_empty_queue_says_so(self, app, monkeypatch):
        app.root.update()
        assert app.empty_label.winfo_ismapped()
        queued(app, monkeypatch, "holiday.jpg")
        app.root.update()
        assert not app.empty_label.winfo_ismapped()

    def test_remove_works_from_what_was_clicked(self, app, monkeypatch):
        queued(app, monkeypatch, "one.jpg", "two.jpg", "three.jpg")
        app._clicked(app.rows[1], click())
        assert app.selected == {1}
        app._clicked(app.rows[2], click(ctrl=True))
        assert app.selected == {1, 2}
        app.remove_selected()
        assert [path.name for path in app.files] == ["one.jpg"]
        assert len(app.rows) == 1 and not app.selected

    def test_the_wheel_turns_it_from_anywhere_over_it(self, app, monkeypatch):
        """Including from over a row, which is where the pointer nearly always
        is -- each row is a widget of its own, and the canvas is behind them."""
        queued(app, monkeypatch, "one.jpg", "two.jpg", "three.jpg")
        turned = []
        monkeypatch.setattr(app.queue_canvas, "yview_scroll",
                            lambda amount, what: turned.append(amount))
        # Standing in for the pointer, which a window nobody has shown cannot
        # have anything under.
        monkeypatch.setattr(app.root, "winfo_containing", lambda *_: app.rows[1].name)
        app._wheeled(SimpleNamespace(x_root=0, y_root=0, delta=-120, num=0))
        app._wheeled(SimpleNamespace(x_root=0, y_root=0, delta=120, num=0))
        assert turned == [2, -2], "down the queue, then back up it"

    def test_but_not_from_off_it(self, app, monkeypatch):
        queued(app, monkeypatch, "one.jpg", "two.jpg", "three.jpg")
        turned = []
        monkeypatch.setattr(app.queue_canvas, "yview_scroll",
                            lambda amount, what: turned.append(amount))
        monkeypatch.setattr(app.root, "winfo_containing", lambda *_: app.canvas)  # the result mat
        app._wheeled(SimpleNamespace(x_root=0, y_root=0, delta=-120, num=0))
        assert not turned

    def test_shift_takes_the_run_between(self, app, monkeypatch):
        queued(app, monkeypatch, "one.jpg", "two.jpg", "three.jpg", "four.jpg")
        app._clicked(app.rows[0], click())
        app._clicked(app.rows[2], click(shift=True))
        assert app.selected == {0, 1, 2}

    def test_the_first_file_puts_up_the_tab_that_governs_it(self, app, monkeypatch):
        queued(app, monkeypatch, "clip.mp4")
        assert app.tabs.tab(app.tabs.select(), "text") == "Video"

    def test_but_a_later_one_does_not_move_it_again(self, app, monkeypatch):
        queued(app, monkeypatch, "holiday.jpg")
        app.tabs.select(0)
        queued(app, monkeypatch, "clip.mp4")
        assert app.tabs.tab(app.tabs.select(), "text") == "General"
