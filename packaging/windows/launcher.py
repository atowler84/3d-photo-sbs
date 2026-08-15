"""Entry point for the packaged Windows build.

One folder holds two exes: `StereoCraft.exe` opens the window, `StereoCraft-cli.exe` is the
command line.  They are the same program -- the analysis PyInstaller does over
Torch is slow and would be identical for both -- so the name the exe was
launched under is what decides which half runs.
"""

import multiprocessing
import os
import sys


def main():
    # A frozen app re-executes itself to make a child process, so anything
    # spawning one has to be told it is already inside the app.
    multiprocessing.freeze_support()

    # Depth Anything 3 puts @torch.jit.script on one small matrix helper, and
    # TorchScript compiles from *source* -- which a frozen app does not have,
    # having shipped .pyc instead.  Turning the JIT off makes the decorator a
    # no-op and the function run as ordinary Python, which for a 4x4 inverse
    # costs nothing.  It has to happen before Torch is imported, which is why it
    # is here and not somewhere more obvious.
    os.environ.setdefault("PYTORCH_JIT", "0")

    # The windowed exe has no console attached, which leaves stdout and stderr
    # as None.  The odd progress line written to either is worth losing, but
    # not worth an AttributeError taking the window down with it.
    null = None
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            null = null or open(os.devnull, "w")
            setattr(sys, name, null)

    if "cli" in os.path.basename(sys.argv[0]).lower():
        from stereocraft.cli import main as run
    else:
        from stereocraft.gui import main as run
    return run()


if __name__ == "__main__":
    sys.exit(main())
