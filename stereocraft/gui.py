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
# The window always uses the best depth model at its best working resolution --
# the only settings on show are the ones that are a matter of taste.
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

        root.title("StereoCraft - side-by-side 3D")
        self._set_icon(root)
        root.minsize(700, 560)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        bar = ttk.Frame(frame)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.add_button = ttk.Button(bar, text="Add files...", command=self.add_files)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(bar, text="Remove", command=self.remove_selected)
        self.remove_button.pack(side="left", padx=4)
        self.clear_button = ttk.Button(bar, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")
        self.dest_label = ttk.Label(bar, text="Saving beside each file")
        self.dest_label.pack(side="right")
        ttk.Button(bar, text="Output folder...", command=self.pick_output).pack(side="right", padx=6)

        # exportselection off, or clicking anything else on the window takes the
        # X selection away and silently empties this one -- which would disable
        # Remove out from under a photo the user can still see highlighted.
        self.listbox = tk.Listbox(frame, height=6, selectmode="extended", activestyle="none",
                                  exportselection=False)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", lambda *_: self._refresh_controls())
        self._row_fg = self.listbox.cget("foreground") or "black"

        self._build_settings(frame)

        self.canvas = tk.Label(frame, background="#1b1b1b", anchor="center")
        self.canvas.grid(row=3, column=0, sticky="nsew", pady=(8, 8))
        frame.rowconfigure(3, weight=2)

        footer = ttk.Frame(frame)
        footer.grid(row=4, column=0, sticky="ew")
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

        # The settings a Reset would undo; watching them is what tells the button
        # whether there is anything left to undo.
        for variable in (self.eyes, self.focus, self.cross, self.automatic):
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

    def _build_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Settings", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
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
        self.save_depth = tk.BooleanVar(value=False)
        self._value_labels = []
        self._manual = []  # the sliders that only apply when not matching the scene

        self._slider(box, 0, "Eye separation", self.eyes, 20.0, 80.0, "{:.0f} mm",
                     "How far apart your two eyes are. 65mm is the human average; smaller is a"
                     " gentler effect, and a video is recommended gentler than a photo.")
        self._slider(box, 2, "Focus distance", self.focus, 1.0 / 20, 1.0 / 0.5,
                     lambda v: f"{1.0 / v:.1f} m",
                     "How far away the window sits. Whatever is at this distance looks like it is"
                     " in the screen; nearer comes out of it, further recedes behind it.")

        ttk.Checkbutton(box, text="Cross-eyed order (only for free-viewing without a headset)",
                        variable=self.cross).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(box, text="Also save the depth map", variable=self.save_depth).grid(
            row=5, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(box, text="Match the scene automatically (recommended)",
                        variable=self.automatic, command=self._refresh_controls).grid(
                            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.reset_button = ttk.Button(box, text="Reset to recommended", command=self.reset_settings)
        self.reset_button.grid(row=7, column=0, sticky="w", pady=(8, 0))

    def _slider(self, parent, row, label, variable, low, high, fmt, hint):
        """`fmt` is a format string, or a callable for a value the slider does not
        hold directly -- focus distance being stored the other way up."""
        show = fmt if callable(fmt) else fmt.format
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        value = ttk.Label(parent, text=show(variable.get()), width=8, anchor="e")
        update = lambda *_: value.config(text=show(variable.get()))
        scale = ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal",
                          command=update)
        scale.grid(row=row, column=1, sticky="ew", padx=8)
        self._manual.append(scale)
        value.grid(row=row, column=2, sticky="w")
        ttk.Label(parent, text=hint, foreground="#777").grid(
            row=row + 1, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._value_labels.append(update)

    def reset_settings(self):
        eyes, focus = self.recommended
        self.eyes.set(eyes)
        self.focus.set(1.0 / focus)
        self.cross.set(False)
        for update in self._value_labels:
            update()

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
        for update in self._value_labels:
            update()
        return wanted

    # --- file list ---------------------------------------------------------
    def add_files(self):
        chosen = filedialog.askopenfilenames(title="Choose photos or videos", filetypes=FILETYPES)
        for name in chosen:
            path = Path(name)
            if path not in self.files:
                self.files.append(path)
                self.states.append(("pending", ""))
                self.listbox.insert("end", self._row_label(len(self.files) - 1))
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

    def _at_defaults(self):
        """Is there anything for Reset to undo?  Only the three settings it
        actually puts back are asked about; saving the depth map is not one."""
        return self._sliders_at(self.recommended) and not self.cross.get()

    def _refresh_controls(self):
        """Offer only what there is currently something to do with."""
        idle = not self.running
        for button, usable in (
            (self.add_button, idle),
            # A batch works from the list as it stood when Convert was pressed,
            # so the list stays put until it has finished with it.
            (self.remove_button, idle and bool(self.listbox.curselection())),
            (self.clear_button, idle and bool(self.files)),
            (self.reset_button, not self._at_defaults()),
            (self.stop_button, self.running and not self.cancel.is_set()),
            *((slider, not self.automatic.get()) for slider in self._manual),
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
        self.cancel.clear()
        self.finished = 0
        self.errors = []
        for index in range(len(self.files)):
            self._set_row(index, "pending")  # a re-run starts everything over
        self._refresh_controls()
        self.progress.config(maximum=len(self.files), value=0)
        automatic = self.automatic.get()
        common = dict(
            eyes_mm="auto" if automatic else round(self.eyes.get(), 1),
            focus_m="auto" if automatic else round(1.0 / self.focus.get(), 3),
            cross_eyed=self.cross.get(),
            on_oversize=self._ask_oversize,
        )
        settings = (Settings(save_depth=self.save_depth.get(), **common), VideoSettings(**common))
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
                                             self._progress(index), self._previewer())
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

    def _previewer(self):
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
                self.events.put(("preview", image))
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

                self.preview = ImageTk.PhotoImage(payload)
                self.canvas.config(image=self.preview)
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
                # A clip has its own last frame on the canvas already; a photo is
                # only ever seen once it is written.
                if "frames" not in info:
                    self._show(info["output"])
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

    def _show(self, path):
        try:
            from PIL import Image, ImageTk

            with Image.open(path) as image:
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH // 2), Image.LANCZOS)
                self.preview = ImageTk.PhotoImage(image.copy())
            self.canvas.config(image=self.preview)
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
