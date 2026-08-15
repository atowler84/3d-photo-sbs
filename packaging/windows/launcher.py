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
