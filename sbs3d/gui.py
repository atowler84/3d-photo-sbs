"""A small desktop window around the converter.

Deliberately plain Tkinter: no extra packages, starts instantly, and keeps the
depth model loaded so the second photo onwards converts in well under a second.
"""

import queue
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

from .pipeline import SUFFIXES, Converter, Settings

FILETYPES = [("Images", " ".join(f"*{s}" for s in sorted(SUFFIXES))), ("All files", "*.*")]
PREVIEW_WIDTH = 620
# The window always uses the best depth model at its best working resolution --
# the only settings on show are the ones that are a matter of taste.
DEFAULT_DISPARITY = Settings.disparity
DEFAULT_CONVERGENCE = Settings.convergence
# How a photo's own row reports on it, so a batch can be followed without
# reading the one status line at the bottom.
MARKS = {"pending": " ", "working": "\u25b6", "done": "\u2713", "skipped": "\u2013", "failed": "\u2715"}
COLOURS = {"done": "#2e7d32", "skipped": "#8a6d1f", "failed": "#b00020", "working": "#1565c0"}


class App:
    def __init__(self, root):
        self.root = root
        self.photos = []
        self.states = []  # one ("state", "detail") per photo, in step with it
        self.running = False
        self.events = queue.Queue()
        self.converter = Converter()
        self.output_dir = None
        self.preview = None
        self.finished = 0
        self.errors = []

        root.title("sbs3d - side-by-side 3D photos")
        root.minsize(700, 560)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        bar = ttk.Frame(frame)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.add_button = ttk.Button(bar, text="Add photos...", command=self.add_photos)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(bar, text="Remove", command=self.remove_selected)
        self.remove_button.pack(side="left", padx=4)
        self.clear_button = ttk.Button(bar, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")
        self.dest_label = ttk.Label(bar, text="Saving beside each photo")
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
        footer.columnconfigure(1, weight=1)
        self.convert_button = ttk.Button(footer, text="Convert", command=self.start)
        self.convert_button.grid(row=0, column=0)
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=8)
        self.status = ttk.Label(footer, text="Add a photo to begin")
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # The settings a Reset would undo; watching them is what tells the button
        # whether there is anything left to undo.
        for variable in (self.disparity, self.convergence, self.cross):
            variable.trace_add("write", lambda *_: self._refresh_controls())
        self._refresh_controls()

        root.after(100, self._drain)

    def _build_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Settings", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)

        self.disparity = tk.DoubleVar(value=DEFAULT_DISPARITY)
        self.convergence = tk.DoubleVar(value=DEFAULT_CONVERGENCE)
        self.cross = tk.BooleanVar(value=False)
        self.save_depth = tk.BooleanVar(value=False)
        self._value_labels = []

        self._slider(box, 0, "Depth strength", self.disparity, 0.5, 4.0, "{:.1f}%",
                     "How far apart your eyes are. Higher is stronger 3D, and harder to look at.")
        self._slider(box, 2, "Screen plane", self.convergence, 0.0, 1.0, "{:.2f}",
                     "Which depth sits at the window. Lower pushes the whole scene further back.")

        ttk.Checkbutton(box, text="Cross-eyed order (only for free-viewing without a headset)",
                        variable=self.cross).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(box, text="Also save the depth map", variable=self.save_depth).grid(
            row=5, column=0, columnspan=3, sticky="w")
        self.reset_button = ttk.Button(box, text="Reset to recommended", command=self.reset_settings)
        self.reset_button.grid(row=6, column=0, sticky="w", pady=(8, 0))

    def _slider(self, parent, row, label, variable, low, high, fmt, hint):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        value = ttk.Label(parent, text=fmt.format(variable.get()), width=7, anchor="e")
        update = lambda *_: value.config(text=fmt.format(variable.get()))
        ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal",
                  command=update).grid(row=row, column=1, sticky="ew", padx=8)
        value.grid(row=row, column=2, sticky="w")
        ttk.Label(parent, text=hint, foreground="#777").grid(
            row=row + 1, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._value_labels.append(update)

    def reset_settings(self):
        self.disparity.set(DEFAULT_DISPARITY)
        self.convergence.set(DEFAULT_CONVERGENCE)
        self.cross.set(False)
        for update in self._value_labels:
            update()

    # --- file list ---------------------------------------------------------
    def add_photos(self):
        chosen = filedialog.askopenfilenames(title="Choose photos", filetypes=FILETYPES)
        for name in chosen:
            path = Path(name)
            if path not in self.photos:
                self.photos.append(path)
                self.states.append(("pending", ""))
                self.listbox.insert("end", self._row_label(len(self.photos) - 1))
        self._refresh_status()

    def remove_selected(self):
        for index in sorted(self.listbox.curselection(), reverse=True):
            self.listbox.delete(index)
            del self.photos[index]
            del self.states[index]
        self._refresh_status()

    def clear(self):
        self.listbox.delete(0, "end")
        self.photos.clear()
        self.states.clear()
        self._refresh_status()

    def _row_label(self, index):
        state, detail = self.states[index]
        label = f"{MARKS[state]}  {self.photos[index].name}"
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
        return (round(self.disparity.get(), 2) == DEFAULT_DISPARITY
                and round(self.convergence.get(), 3) == DEFAULT_CONVERGENCE
                and not self.cross.get())

    def _refresh_controls(self):
        """Offer only what there is currently something to do with."""
        idle = not self.running
        for button, usable in (
            (self.add_button, idle),
            # A batch works from the list as it stood when Convert was pressed,
            # so the list stays put until it has finished with it.
            (self.remove_button, idle and bool(self.listbox.curselection())),
            (self.clear_button, idle and bool(self.photos)),
            (self.reset_button, not self._at_defaults()),
        ):
            button.state(["!disabled"] if usable else ["disabled"])

    def pick_output(self):
        folder = filedialog.askdirectory(title="Choose an output folder")
        self.output_dir = Path(folder) if folder else None
        self.dest_label.config(text=f"Saving to {self.output_dir.name}" if self.output_dir
                               else "Saving beside each photo")

    def _refresh_status(self):
        count = len(self.photos)
        self.status.config(text="Add a photo to begin" if not count
                           else f"{count} photo{'s' if count > 1 else ''} ready")
        self._refresh_controls()

    # --- conversion --------------------------------------------------------
    def start(self):
        if not self.photos:
            self.add_photos()
            return
        self.convert_button.state(["disabled"])
        self.running = True
        self.finished = 0
        self.errors = []
        for index in range(len(self.photos)):
            self._set_row(index, "pending")  # a re-run starts everything over
        self._refresh_controls()
        self.progress.config(maximum=len(self.photos), value=0)
        settings = Settings(
            disparity=round(self.disparity.get(), 2),
            convergence=round(self.convergence.get(), 3),
            cross_eyed=self.cross.get(),
            save_depth=self.save_depth.get(),
            on_oversize=self._ask_oversize,
        )
        self.converter.settings = settings
        threading.Thread(target=self._work, args=(list(self.photos), self.output_dir), daemon=True).start()

    def _ask_oversize(self, oversize):
        """Put a too-large photo to the user.  Runs on the worker thread, so the
        question goes to the main loop as an event and waits for the answer."""
        reply = queue.Queue(maxsize=1)
        self.events.put(("ask", (oversize, reply)))
        return reply.get()

    def _work(self, photos, output_dir):
        try:
            self.events.put(("status", "Loading depth model..."))
            try:
                self.converter.depth_model  # pay the load cost before the first photo
            except Exception as error:
                self.events.put(("error", (None, f"Could not load the depth model: {error}")))
                return
            for index, photo in enumerate(photos):
                self.events.put(("status", f"Converting {photo.name} ({index + 1}/{len(photos)})"))
                self.events.put(("working", index))
                try:
                    info = self.converter.convert(photo, output_dir)
                except Exception as error:
                    self.events.put(("error", (index, f"{photo.name}: {error}")))
                    continue
                if info is None:  # too large, and the user chose to skip it
                    self.events.put(("skipped", (index, photo)))
                    continue
                self.events.put(("done", (index, info)))
        finally:
            self.events.put(("finished", None))

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
                self._set_row(index, "done",
                              f"{width}x{height} in {info['seconds']:.1f}s"
                              + (" (resized)" if was else ""))
                self.status.config(
                    text=f"{info['output'].name}  -  {width}x{height}  "
                         f"in {info['seconds']:.1f}s{note}")
                self._show(info["output"])
            elif kind == "finished":
                self.convert_button.state(["!disabled"])
                self.running = False
                self._refresh_controls()
                if self.errors:
                    messagebox.showerror("sbs3d", "\n".join(self.errors))
        self.root.after(100, self._drain)

    def _oversize_dialog(self, oversize):
        """The modal question itself, on the main thread where Tk wants it."""
        if oversize.target is None:
            messagebox.showerror("sbs3d", oversize.describe())
            return "skip"
        resize = messagebox.askyesno(
            "sbs3d - photo too large",
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
