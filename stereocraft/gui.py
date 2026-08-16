"""A small desktop window around the converter.

Deliberately plain Tkinter: no extra packages, starts instantly, and keeps the
depth model loaded so the second photo onwards converts in well under a second.
"""

import queue
import sys
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

from .pipeline import SUFFIXES, Converter, Settings, VideoSettings
from .video import VIDEO_SUFFIXES, clock, convert_video

FILETYPES = [
    ("Photos and videos", " ".join(f"*{s}" for s in sorted(SUFFIXES | VIDEO_SUFFIXES))),
    ("Photos", " ".join(f"*{s}" for s in sorted(SUFFIXES))),
    ("Videos", " ".join(f"*{s}" for s in sorted(VIDEO_SUFFIXES))),
    ("All files", "*.*"),
]
PREVIEW_WIDTH = 620
# How wide the explanatory line under a setting is allowed to run before it
# wraps, which is what stops a long hint deciding how wide the window is.
HINT_WIDTH = 560
# The result sits on a dark mat, the way a print is mounted -- it is what a
# side-by-side pair is easiest to judge against, and it stops a bright photo
# glaring against the window behind it.
MAT = "#1b1b1b"
MAT_TEXT = "#9a9a9a"
EMPTY = ("Nothing converted yet\n\n"
         "The side-by-side pair appears here as each file finishes,\n"
         "and frame by frame while a clip is being converted.")
NO_CAPTION = " "  # keeps the caption's line of height, so nothing jumps later
# The cap on the finished pair's width, for a viewer that will not open
# something enormous.  Native is no cap at all, which is the usual answer.
WIDTHS = [("Native size", 0), ("Up to 4096 px", 4096), ("Up to 6144 px", 6144),
          ("Up to 8192 px", 8192), ("Up to 12288 px", 12288)]
# The window always uses the best depth model at its best working resolution, and
# neither is on show: what is on show is what the picture looks like and what it
# comes out as, which is what there is any judgement in.
#
# What is recommended depends on whether the picture moves: a clip wants a
# gentler depth than a still, because an error the eye forgives in something it
# glances at becomes a shimmer it cannot ignore over several minutes.
# Where the manual sliders sit when they are not being driven by the scene.
# `Settings` itself defaults to matching the scene, so these are the starting
# points for someone who has turned that off: a real pair of eyes at a
# comfortable distance, and something gentler once the picture moves.
RECOMMENDED = {False: (65.0, 3.0), True: (45.0, 3.0)}
# How often the window redraws the frame a conversion is currently on.  Often
# enough to look live, seldom enough that Tk is not the slowest part of it.
PREVIEW_EVERY = 1.0


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


# How each file's own row reports on it, so a batch can be followed without
# reading the one status line at the bottom.
MARKS = {"pending": " ", "working": "\u25b6", "done": "\u2713", "skipped": "\u2013",
         "failed": "\u2715", "stopped": "\u2013"}
COLOURS = {"done": "#2e7d32", "skipped": "#8a6d1f", "failed": "#b00020", "working": "#1565c0",
           "stopped": "#8a6d1f"}


class App:
    def __init__(self, root):
        self.root = root
        self.files = []
        self.states = []  # one ("state", "detail") per file, in step with it
        self.running = False
        self.cancel = threading.Event()
        # What the sliders would say if nobody had touched them, which depends on
        # what is in the queue; see `_sync_recommendation`.
        self.recommended = RECOMMENDED[False]
        self.events = queue.Queue()
        self.converter = Converter()
        self.output_dir = None
        self.preview = None
        self.finished = 0
        self.errors = []
        self.cross_used = False  # the viewing order the current run is writing

        root.title("StereoCraft - side-by-side 3D")
        self._set_icon(root)
        root.minsize(960, 720)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        # The queue keeps a fixed width and everything else takes the slack:
        # file names are all much of a length, whereas the result is what has
        # something to do with every extra pixel.
        frame.columnconfigure(0, minsize=250)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self._build_queue(frame)
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.columnconfigure(0, weight=1)
        self._build_settings(right)
        self._build_result(right)

        footer = ttk.Frame(frame)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        footer.columnconfigure(2, weight=1)
        self.convert_button = ttk.Button(footer, text="Convert", command=self.start)
        self.convert_button.grid(row=0, column=0)
        # A photo is over before anyone could ask for it back; a clip runs for
        # minutes, so there has to be a way out of one.
        self.stop_button = ttk.Button(footer, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=(6, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=8)
        self.status = ttk.Label(footer, text="Add a photo or a video to begin")
        self.status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # Every setting a Reset would put back.  Watching all of them is what
        # tells each tab's button whether it has anything left to undo, and the
        # eyes and the focus which of them are the user's to set.
        for variable in (self.automatic, self.eyes, self.focus, self.cross,
                         self.photo_depth, self.fmt, self.quality, self.max_size,
                         self.save_depth, self.target, self.temporal, self.crf,
                         self.codec, self.audio, self.full_width):
            variable.trace_add("write", lambda *_: self._refresh_controls())
        self._refresh_controls()

        root.after(100, self._drain)

    @staticmethod
    def _set_icon(root):
        """Put the app's icon on the window.

        Tk draws its own title bar icon and defaults to the Tcl feather, so the
        one built into the exe never reaches the window and has to be set here.
        """
        path = Path(__file__).with_name("stereocraft.ico")
        if not path.exists():  # running from a checkout without the icon
            return
        try:
            root.iconbitmap(default=str(path))  # Windows takes the .ico itself
        except tk.TclError:
            try:  # X11 wants an image rather than an .ico, so decode it first
                from PIL import Image, ImageTk

                with Image.open(path) as image:
                    root._icon = ImageTk.PhotoImage(image)  # Tk drops it unreferenced
                root.iconphoto(True, root._icon)
            except Exception:  # a window with the wrong icon still converts photos
                pass

    def _build_queue(self, parent):
        """The list of files, down the left-hand side of the window.

        A column rather than a band across the top: a queue grows downwards, so
        given the height it shows a whole batch at once, and the width it gives
        up is width the picture beside it would not have used anyway.
        """
        box = ttk.LabelFrame(parent, text="Queue", padding=8)
        box.grid(row=0, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)

        buttons = ttk.Frame(box)
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.add_button = ttk.Button(buttons, text="Add files...", command=self.add_files)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(buttons, text="Remove", command=self.remove_selected)
        self.remove_button.pack(side="left", padx=4)
        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")

        rows = ttk.Frame(box)
        rows.grid(row=1, column=0, sticky="nsew")
        rows.columnconfigure(0, weight=1)
        rows.rowconfigure(0, weight=1)
        # exportselection off, or clicking anything else on the window takes the
        # X selection away and silently empties this one -- which would disable
        # Remove out from under a photo the user can still see highlighted.
        self.listbox = tk.Listbox(rows, width=28, height=12, selectmode="extended",
                                  activestyle="none", exportselection=False,
                                  borderwidth=1, relief="solid", highlightthickness=0)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        # A tall list is a list worth scrolling, and the bar only appears once
        # there is more in the queue than fits.
        scroll = ttk.Scrollbar(rows, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=lambda first, last: self._scrollbar(scroll, first, last))
        self.listbox.bind("<<ListboxSelect>>", lambda *_: self._refresh_controls())
        self._row_fg = self.listbox.cget("foreground") or "black"

        ttk.Button(box, text="Output folder...", command=self.pick_output).grid(
            row=2, column=0, sticky="w", pady=(8, 0))
        self.dest_label = ttk.Label(box, text="Saving beside each file", foreground="#777",
                                    wraplength=220)
        self.dest_label.grid(row=3, column=0, sticky="w", pady=(2, 0))

    @staticmethod
    def _scrollbar(scroll, first, last):
        """Show the queue's scrollbar only while it has somewhere to scroll."""
        scroll.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            scroll.grid_remove()
        else:
            scroll.grid(row=0, column=1, sticky="ns", padx=(2, 0))

    def _build_settings(self, parent):
        """The settings, as three tabs of what they apply to.

        Split that way rather than by what they do: a run is usually all photos
        or all clips, and a tab puts the half that has nothing to say about the
        queue out of the way instead of greying it out in front of you.
        """
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=0, column=0, sticky="ew")
        # One Reset per tab, each with the question of whether it has anything
        # left to put back; see `_reset_button`.
        self._resets = []
        self._manual = []  # sliders that only apply when NOT matching the scene
        self._auto_only = []  # ...and the ones that only apply when it is
        self._build_general(self.tabs)
        self.photo_tab = self._build_photo(self.tabs)
        self.video_tab = self._build_video(self.tabs)

    def _build_general(self, tabs):
        """What the scene looks like, whether it moves or not."""
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="General")
        box.columnconfigure(1, weight=1)

        # On by default, because sizing the eyes to the scene beats any single
        # number: the same 65mm that suits a portrait renders a close-up
        # unfusable and a telephoto shot flat.
        self.automatic = tk.BooleanVar(value=True)
        self.eyes = tk.DoubleVar(value=self.recommended[0])
        # Held as 1/metres rather than metres.  That is the quantity the geometry
        # is linear in, so an inch of travel changes the picture by as much at one
        # end of the slider as at the other; in metres the far half would do
        # almost nothing and the near half everything.
        self.focus = tk.DoubleVar(value=1.0 / self.recommended[1])
        self.cross = tk.BooleanVar(value=False)

        # First, because it decides which of the two below it are yours to set:
        # matching the scene drives the eyes and the focus itself, and takes its
        # instructions from the Depth slider on the tab for what is in the queue.
        ttk.Checkbutton(box, text="Match the scene automatically (recommended)",
                        variable=self.automatic).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="Sizes the eyes and the focus to what is actually in the picture,"
                            " and aims for the Depth set on the Photo or Video tab.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self._manual.append(self._slider(
            box, 2, "Eye separation", self.eyes, 20.0, 80.0, "{:.0f} mm",
            "How far apart your two eyes are. 65mm is the human average; smaller is a"
            " gentler effect, and a video is recommended gentler than a photo."))
        self._manual.append(self._slider(
            box, 4, "Focus distance", self.focus, 1.0 / 20, 1.0 / 0.5,
            lambda v: f"{1.0 / v:.1f} m",
            "How far away the window sits. Whatever is at this distance looks like it is"
            " in the screen; nearer comes out of it, further recedes behind it."))

        ttk.Checkbutton(box, text="Cross-eyed order (only for free-viewing without a headset)",
                        variable=self.cross).grid(row=6, column=0, columnspan=3, sticky="w",
                                                  pady=(4, 0))
        self._reset_button(box, 7, self.reset_general, self._general_at_default)

    def _build_photo(self, tabs):
        """How much depth a still asks for, and what it is written as."""
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="Photo")
        box.columnconfigure(1, weight=1)

        self.photo_depth = tk.DoubleVar(value=Settings.target_pct)
        self.fmt = tk.StringVar(value=Settings.fmt)
        self.quality = tk.DoubleVar(value=Settings.quality)
        self.max_size = tk.StringVar(value=WIDTHS[0][0])
        self.save_depth = tk.BooleanVar(value=False)

        self._auto_only.append(self._slider(
            box, 0, "Depth", self.photo_depth, 0.5, 3.5, "{:.1f}%",
            "How much separation matching the scene aims for, as a share of the width."
            " 2% is a comfortable pair of eyes; more is stronger and harder to fuse."))

        row = ttk.Frame(box)
        row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="File format").pack(side="left")
        for value, text in (("auto", "Same as the photo"), ("jpg", "JPEG"), ("png", "PNG")):
            ttk.Radiobutton(row, text=text, value=value,
                            variable=self.fmt).pack(side="left", padx=(8, 0))
        ttk.Label(box, text="A side-by-side pair is twice the pixels of the photo, so JPEG at a"
                            " high quality is usually the sensible one. PNG is lossless and large.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self.quality_scale = self._slider(
            box, 4, "JPEG quality", self.quality, 70.0, 100.0, "{:.0f}",
            "95 keeps the compression out of the depth cues; below about 85 the edges the"
            " warp opened up start to show as blocks.")

        row = ttk.Frame(box)
        row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="Output width").pack(side="left")
        self.width_box = ttk.Combobox(row, textvariable=self.max_size, state="readonly", width=18,
                                      values=[name for name, _ in WIDTHS])
        self.width_box.pack(side="left", padx=(8, 0))
        ttk.Label(box, text="A cap for a viewer that will not open something enormous. The pair"
                            " comes out about twice as wide as the photo went in.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Checkbutton(box, text="Also save the depth map", variable=self.save_depth).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._reset_button(box, 9, self.reset_photo, self._photo_at_default)
        return box

    def _build_video(self, tabs):
        """What a clip costs to look at for several minutes.

        Its own tab rather than mixed in with the two sliders on General: those
        are about what the scene looks like, these are about what holds up over
        a few thousand frames of it.
        """
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="Video")
        box.columnconfigure(1, weight=1)

        self.target = tk.DoubleVar(value=VideoSettings.target_pct)
        self.temporal = tk.DoubleVar(value=VideoSettings.temporal)
        self.codec = tk.StringVar(value=VideoSettings.codec)
        self.crf = tk.DoubleVar(value=VideoSettings.crf)
        self.audio = tk.BooleanVar(value=VideoSettings.audio)
        self.full_width = tk.BooleanVar(value=VideoSettings.full_width)

        self._auto_only.append(self._slider(
            box, 0, "Depth", self.target, 0.4, 3.0, "{:.1f}%",
            "How much separation a clip aims for. Lower than a photo on purpose:"
            " an error you would not notice in a still shimmers once it moves."))
        self._slider(box, 2, "Steadiness", self.temporal, 0.0, 0.95, "{:.2f}",
                     "How much of each frame's depth carries into the next. Higher is"
                     " calmer and very slightly softer; 0 turns it off.")
        self._slider(box, 4, "Encoder quality", self.crf, 14.0, 28.0, "CRF {:.0f}",
                     "Lower is better and bigger. 18 is visually lossless; 23 is about half"
                     " the file and still good.")

        row = ttk.Frame(box)
        row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="Codec").pack(side="left")
        for value, text in (("h264", "H.264 (plays anywhere)"), ("hevc", "HEVC (better above 4K)")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.codec).pack(side="left", padx=(8, 0))

        ttk.Checkbutton(box, text="Keep the soundtrack", variable=self.audio).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(box, text="Full width (every native pixel, twice as wide)",
                        variable=self.full_width).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="Off squeezes each eye to half width, which is what players and"
                            " headsets expect and what their decoders can keep up with.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=9, column=0, columnspan=3, sticky="w")
        self._reset_button(box, 10, self.reset_video, self._video_at_default)
        return box

    def _build_result(self, parent):
        """Where the finished pair is shown.

        A picture wants a dark mat around it, but a dark rectangle with nothing
        in it reads as something broken -- so until there is a picture the mat
        says what will be arriving in it, and afterwards a line underneath says
        what it is looking at.
        """
        box = ttk.LabelFrame(parent, text="Result", padding=8)
        box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1, minsize=PREVIEW_WIDTH // 2 + 16)

        self.canvas = tk.Label(box, background=MAT, foreground=MAT_TEXT, text=EMPTY,
                               justify="center", anchor="center", compound="center",
                               borderwidth=1, relief="solid", padx=10, pady=10)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.caption = ttk.Label(box, text=NO_CAPTION, foreground="#777", anchor="center")
        self.caption.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def _slider(self, parent, row, label, variable, low, high, fmt, hint):
        """`fmt` is a format string, or a callable for a value the slider does not
        hold directly -- focus distance being stored the other way up."""
        show = fmt if callable(fmt) else fmt.format
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        value = ttk.Label(parent, text=show(variable.get()), width=8, anchor="e")
        # Watching the variable rather than the slider, so a value put back by a
        # Reset shows up the same as one dragged there.
        variable.trace_add("write", lambda *_: value.config(text=show(variable.get())))
        scale = ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal")
        scale.grid(row=row, column=1, sticky="ew", padx=8)
        value.grid(row=row, column=2, sticky="w")
        ttk.Label(parent, text=hint, foreground="#777", wraplength=HINT_WIDTH,
                  justify="left").grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=(0, 4))
        return scale  # so the caller can say when it applies; see `_refresh_controls`

    def _reset_button(self, parent, row, reset, at_default):
        """Each tab puts its own settings back, and only its own.

        One button for the lot would be a button that undoes work on a tab you
        cannot see; and it goes grey when its tab is already at its defaults,
        which is also how you can tell at a glance that it is.
        """
        button = ttk.Button(parent, text="Reset to default", command=reset)
        button.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self._resets.append((button, at_default))
        return button

    def reset_general(self):
        eyes, focus = self.recommended
        self.automatic.set(True)
        self.eyes.set(eyes)
        self.focus.set(1.0 / focus)
        self.cross.set(False)

    def _general_at_default(self):
        """What the eyes and the focus go back to follows the queue: a clip is
        recommended gentler than a still.  See `_sync_recommendation`."""
        return (self.automatic.get() and self._sliders_at(self.recommended)
                and not self.cross.get())

    def reset_photo(self):
        self.photo_depth.set(Settings.target_pct)
        self.fmt.set(Settings.fmt)
        self.quality.set(Settings.quality)
        self.max_size.set(WIDTHS[0][0])
        self.save_depth.set(False)

    def _photo_at_default(self):
        return (round(self.photo_depth.get(), 2) == Settings.target_pct
                and self.fmt.get() == Settings.fmt
                and round(self.quality.get()) == Settings.quality
                and self.max_size.get() == WIDTHS[0][0]
                and not self.save_depth.get())

    def reset_video(self):
        self.target.set(VideoSettings.target_pct)
        self.temporal.set(VideoSettings.temporal)
        self.crf.set(VideoSettings.crf)
        self.codec.set(VideoSettings.codec)
        self.audio.set(VideoSettings.audio)
        self.full_width.set(VideoSettings.full_width)

    def _video_at_default(self):
        return (round(self.target.get(), 2) == VideoSettings.target_pct
                and round(self.temporal.get(), 2) == VideoSettings.temporal
                and round(self.crf.get()) == VideoSettings.crf
                and self.codec.get() == VideoSettings.codec
                and self.audio.get() == VideoSettings.audio
                and self.full_width.get() == VideoSettings.full_width)

    def _sliders_at(self, values):
        eyes, focus = values
        return (round(self.eyes.get(), 1) == round(eyes, 1)
                and round(1.0 / self.focus.get(), 2) == round(focus, 2))

    def _sync_recommendation(self):
        """Follow the queue: a video recommends gentler settings than a photo.

        Sliders still sitting where the app left them are the app's to move, and
        move.  Sliders the user has set are theirs, and are left alone -- they
        only find out what changed from the note at the bottom.  A queue holding
        both takes the video's advice, being the more conservative of the two.
        """
        wanted = RECOMMENDED[any(is_video(path) for path in self.files)]
        if wanted == self.recommended:
            return None
        untouched = self._sliders_at(self.recommended)
        self.recommended = wanted
        if not untouched:
            return None
        self.eyes.set(wanted[0])
        self.focus.set(1.0 / wanted[1])
        return wanted

    # --- file list ---------------------------------------------------------
    def add_files(self):
        chosen = filedialog.askopenfilenames(title="Choose photos or videos", filetypes=FILETYPES)
        first = not self.files
        for name in chosen:
            path = Path(name)
            if path not in self.files:
                self.files.append(path)
                self.states.append(("pending", ""))
                self.listbox.insert("end", self._row_label(len(self.files) - 1))
        # Opening a queue puts up the tab that has something to say about it.
        # Only the first time, so a tab chosen since is not taken away again.
        if first and self.files:
            self.tabs.select(self.video_tab if is_video(self.files[0]) else self.photo_tab)
        moved = self._sync_recommendation()
        self._refresh_status()
        if moved:
            self.status.config(text=f"{self.status.cget('text')}  -  eyes eased to "
                                    f"{moved[0]:.0f}mm for video")

    def remove_selected(self):
        for index in sorted(self.listbox.curselection(), reverse=True):
            self.listbox.delete(index)
            del self.files[index]
            del self.states[index]
        self._sync_recommendation()
        self._refresh_status()

    def clear(self):
        self.listbox.delete(0, "end")
        self.files.clear()
        self.states.clear()
        self._sync_recommendation()
        self._refresh_status()
        self._clear_result()

    def _row_label(self, index):
        state, detail = self.states[index]
        label = f"{MARKS[state]}  {self.files[index].name}"
        return f"{label}   {detail}" if detail else label

    def _set_row(self, index, state, detail=""):
        """Rewrite one row of the queue.

        A Listbox cannot have a line changed in place, so it goes out and comes
        back -- and takes its selection with it, which the user would otherwise
        watch vanish as their batch runs.
        """
        self.states[index] = (state, detail)
        selected = index in self.listbox.curselection()
        self.listbox.delete(index)
        self.listbox.insert(index, self._row_label(index))
        self.listbox.itemconfig(index, foreground=COLOURS.get(state, self._row_fg))
        if selected:
            self.listbox.selection_set(index)
        if state == "working":
            self.listbox.see(index)  # follow a long batch down the list

    def _refresh_controls(self):
        """Offer only what there is currently something to do with."""
        idle = not self.running
        for button, usable in (
            (self.add_button, idle),
            # A batch works from the list as it stood when Convert was pressed,
            # so the list stays put until it has finished with it.
            (self.remove_button, idle and bool(self.listbox.curselection())),
            (self.clear_button, idle and bool(self.files)),
            *((button, not at_default()) for button, at_default in self._resets),
            (self.stop_button, self.running and not self.cancel.is_set()),
            # The eyes and the focus are yours only when the scene is not
            # setting them; the Depth sliders are the other way round, being
            # what matching the scene aims for.
            *((slider, not self.automatic.get()) for slider in self._manual),
            *((slider, self.automatic.get()) for slider in self._auto_only),
            (self.quality_scale, self.fmt.get() != "png"),
        ):
            button.state(["!disabled"] if usable else ["disabled"])

    def pick_output(self):
        folder = filedialog.askdirectory(title="Choose an output folder")
        self.output_dir = Path(folder) if folder else None
        self.dest_label.config(text=f"Saving to {self.output_dir.name}" if self.output_dir
                               else "Saving beside each file")

    def _refresh_status(self):
        count = len(self.files)
        clips = sum(is_video(path) for path in self.files)
        if not count:
            what = "Add a photo or a video to begin"
        elif clips == count:
            what = f"{count} video{'s' if count > 1 else ''} ready"
        elif clips:
            what = f"{count - clips} photo{'s' if count - clips > 1 else ''} and {clips} " \
                   f"video{'s' if clips > 1 else ''} ready"
        else:
            what = f"{count} photo{'s' if count > 1 else ''} ready"
        self.status.config(text=what)
        self._refresh_controls()

    # --- conversion --------------------------------------------------------
    def start(self):
        if not self.files:
            self.add_files()
            return
        self.convert_button.state(["disabled"])
        self.running = True
        self.cross_used = self.cross.get()
        self.cancel.clear()
        self.finished = 0
        self.errors = []
        for index in range(len(self.files)):
            self._set_row(index, "pending")  # a re-run starts everything over
        self._refresh_controls()
        self._clear_result("Converting...")
        self.progress.config(maximum=len(self.files), value=0)
        automatic = self.automatic.get()
        common = dict(
            eyes_mm="auto" if automatic else round(self.eyes.get(), 1),
            focus_m="auto" if automatic else round(1.0 / self.focus.get(), 3),
            cross_eyed=self.cross.get(),
            on_oversize=self._ask_oversize,
        )
        settings = (Settings(target_pct=round(self.photo_depth.get(), 2),
                             quality=int(round(self.quality.get())), fmt=self.fmt.get(),
                             max_size=dict(WIDTHS)[self.max_size.get()],
                             save_depth=self.save_depth.get(), **common),
                    VideoSettings(target_pct=round(self.target.get(), 2),
                                  temporal=round(self.temporal.get(), 2),
                                  crf=int(round(self.crf.get())), codec=self.codec.get(),
                                  audio=self.audio.get(), full_width=self.full_width.get(),
                                  **common))
        threading.Thread(target=self._work, args=(list(self.files), self.output_dir, settings),
                         daemon=True).start()

    def stop(self):
        """Ask the run to stop.  A clip gives up on the frame it is on and leaves
        no half-written file; a batch of photos stops after the one in hand."""
        self.cancel.set()
        self.status.config(text="Stopping...")
        self._refresh_controls()

    def _ask_oversize(self, oversize):
        """Put a too-large photo to the user.  Runs on the worker thread, so the
        question goes to the main loop as an event and waits for the answer."""
        reply = queue.Queue(maxsize=1)
        self.events.put(("ask", (oversize, reply)))
        return reply.get()

    def _work(self, files, output_dir, settings):
        for_photos, for_videos = settings
        try:
            self.events.put(("status", "Loading depth model..."))
            try:
                self.converter.settings = for_photos
                self.converter.depth_model  # pay the load cost before the first file
            except Exception as error:
                self.events.put(("error", (None, f"Could not load the depth model: {error}")))
                return
            for index, path in enumerate(files):
                if self.cancel.is_set():
                    self.events.put(("stopped", index))
                    continue
                self.events.put(("status", f"Converting {path.name} ({index + 1}/{len(files)})"))
                self.events.put(("working", index))
                moving = is_video(path)
                self.converter.settings = for_videos if moving else for_photos
                try:
                    if moving:
                        info = convert_video(path, output_dir, self.converter,
                                             self._progress(index), self._previewer(path))
                    else:
                        info = self.converter.convert(path, output_dir)
                except Exception as error:
                    self.events.put(("error", (index, f"{path.name}: {error}")))
                    continue
                if info is None:  # too large and skipped, or stopped part-way
                    kind = "stopped" if self.cancel.is_set() else "skipped"
                    self.events.put((kind, index if kind == "stopped" else (index, path)))
                    continue
                self.events.put(("done", (index, info)))
        finally:
            self.events.put(("finished", None))

    def _progress(self, index):
        """Report a clip's frames back to the window, and carry the Stop button's
        answer back to the conversion."""
        warm = 0.0

        def report(done, total, seconds):
            nonlocal warm
            if self.cancel.is_set():
                return False
            # The first frame carries the graphics driver's warm-up with it, and
            # charging the whole clip for it makes the opening estimate nonsense.
            if done == 1:
                warm = seconds
            rate = (done - 1) / (seconds - warm) if done > 1 and seconds > warm else 0
            left = clock((total - done) / rate) if rate and total > done else ""
            self.events.put(("frame", (index, done, total, left)))
            return True

        return report

    def _previewer(self, path):
        """Show the frame being worked on, now and then.

        The array has already been made for the encoder, so this costs the
        shrinking and nothing else -- and it is done here on the worker thread,
        leaving the window with an image it only has to draw.
        """
        last = 0.0

        def show(pixels):
            nonlocal last
            now = time.monotonic()
            if now - last < PREVIEW_EVERY:
                return
            last = now
            try:
                from PIL import Image

                image = Image.fromarray(pixels)
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH // 2), Image.BILINEAR)
                self.events.put(("preview", (image, f"{path.name} - the frame being converted")))
            except Exception:  # a preview is a nicety; the encode carries on
                pass

        return show

    def _drain(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.config(text=payload)
            elif kind == "working":
                self._set_row(payload, "working", "converting...")
            elif kind == "error":
                # Collected rather than popped one dialog at a time, so a long
                # batch cannot bury the user in modal windows.
                index, message = payload
                self.errors.append(message)
                self.status.config(text=message)
                if index is not None:
                    # The row keeps the gist; the dialog at the end has it all.
                    reason = message.split(": ", 1)[-1]
                    self._set_row(index, "failed",
                                  reason if len(reason) <= 60 else reason[:57] + "...")
            elif kind == "ask":
                oversize, reply = payload
                reply.put(self._oversize_dialog(oversize))
            elif kind == "frame":
                index, done, total, left = payload
                share = f"{done}/{total}" if total else f"{done}"
                self._set_row(index, "working",
                              f"frame {share}{f' - {left} left' if left else ''}")
                # The bar runs across the whole queue, and a clip advances it a
                # fraction of a file at a time rather than jumping at the end.
                if total:
                    self.progress.config(value=self.finished + done / total)
            elif kind == "preview":
                from PIL import ImageTk

                image, caption = payload
                self._show_result(ImageTk.PhotoImage(image), caption)
            elif kind == "stopped":
                self.finished += 1
                self.progress.config(value=self.finished)
                self._set_row(payload, "stopped", "stopped")
            elif kind == "skipped":
                index, photo = payload
                self.finished += 1
                self.progress.config(value=self.finished)
                self._set_row(index, "skipped", "too large - skipped")
                self.status.config(text=f"Skipped {photo.name} - too large for memory")
            elif kind == "done":
                index, info = payload
                self.finished += 1
                self.progress.config(value=self.finished)  # step() wraps to 0 at the end
                width, height = info["output_size"]
                was = info["resized_from"]
                note = f"  (resized from {was[0]}x{was[1]})" if was else ""
                if "frames" in info:
                    detail = f"{width}x{height}, {info['frames']} frames in {clock(info['seconds'])}"
                else:
                    detail = f"{width}x{height} in {info['seconds']:.1f}s" + (" (resized)" if was else "")
                self._set_row(index, "done", detail)
                self.status.config(text=f"{info['output'].name}  -  {detail}{note}")
                # A clip has its own last frame on the mat already, so it only
                # needs the caption saying what that frame turned out to be; a
                # photo is not seen at all until it is written.
                # The line under the picture says what it is and how to look
                # at it, the two things the file itself cannot tell you; the
                # status line below already has the timings.
                order = "cross-eyed order" if self.cross_used else "left eye on the left"
                caption = f"{info['output'].name}  -  {width}x{height}, {order}"
                if "frames" in info:
                    self.caption.config(text=caption)
                else:
                    self._show(info["output"], caption)
            elif kind == "finished":
                self.convert_button.state(["!disabled"])
                self.running = False
                if self.cancel.is_set():
                    self.status.config(text="Stopped")
                self.cancel.clear()
                self._refresh_controls()
                if self.errors:
                    messagebox.showerror("StereoCraft", "\n".join(self.errors))
        self.root.after(100, self._drain)

    def _oversize_dialog(self, oversize):
        """The modal question itself, on the main thread where Tk wants it."""
        if oversize.target is None:
            messagebox.showerror("StereoCraft", oversize.describe())
            return "skip"
        resize = messagebox.askyesno(
            "StereoCraft - photo too large",
            f"{oversize.describe()}\n\nResize it and convert, or skip this photo?",
            default="yes", icon="question")
        return "resize" if resize else "skip"

    def _show_result(self, image, caption):
        """Put a picture on the mat, in place of whatever it was saying."""
        self.preview = image  # Tk drops an image nothing is holding on to
        self.canvas.config(image=image, text="")
        self.caption.config(text=caption)

    def _clear_result(self, text=EMPTY):
        """Take the picture off it again, leaving something to read instead."""
        self.preview = None
        self.canvas.config(image="", text=text)
        self.caption.config(text=NO_CAPTION)

    def _show(self, path, caption):
        try:
            from PIL import Image, ImageTk

            with Image.open(path) as image:
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH // 2), Image.LANCZOS)
                self._show_result(ImageTk.PhotoImage(image.copy()), caption)
        except Exception:  # a preview is a nicety; the file is already written
            pass


def main():
    if tk is None:
        print("The desktop window needs Tkinter, which this Python was built without.\n"
              "On Debian/Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        return 1
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
