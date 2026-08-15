"""Command line entry point."""

import argparse
import glob
import sys
from pathlib import Path

from . import __version__
from .pipeline import SUFFIXES, Converter, Settings


def collect(inputs):
    """Expand files, directories and globs into a sorted list of photos."""
    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found += sorted(p for p in path.iterdir() if p.suffix.lower() in SUFFIXES)
        elif path.exists():
            found.append(path)
        else:  # let the shell off the hook for unexpanded globs
            matches = sorted(Path(match) for match in glob.glob(item))
            if not matches:
                print(f"skipping {item}: not found", file=sys.stderr)
            found += matches
    return [p for p in found if not p.stem.endswith(("_sbs", "_depth"))]


def oversize_handler(mode):
    """What to do about a photo that will not fit, as `Settings.on_oversize`.

    Asking needs someone there to answer, so a pipe or a cron job quietly gets
    the safe half of the choice rather than blocking forever on a prompt.
    """
    def settled(oversize):
        # The outcome line covers an ordinary resize; a photo no resize can
        # save still needs explaining.
        if oversize.target is None:
            print(f"\n{oversize.describe()}", file=sys.stderr)
        return mode

    if mode in ("resize", "skip"):
        return settled

    def ask(oversize):
        print(f"\n{oversize.describe()}", file=sys.stderr)
        if oversize.target is None:
            return "skip"  # nothing to offer, and describe() has said why
        if not sys.stdin.isatty():
            print("Not running interactively, so skipping it. "
                  "Pass --oversize resize to shrink photos like this instead.", file=sys.stderr)
            return "skip"
        while True:
            answer = input("Resize and convert it, or skip it? [r/s] ").strip().lower()
            if answer in ("r", "resize"):
                return "resize"
            if answer in ("s", "skip", ""):
                return "skip"

    return ask


class Defaults(argparse.ArgumentDefaultsHelpFormatter):
    """The stock formatter, less the "(default: None)" it would otherwise print
    against the two settings whose default depends on whether the input moves."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stereocraft",
        description="Turn a photo into a full-resolution side-by-side 3D image.",
        formatter_class=Defaults,
    )
    parser.add_argument("inputs", nargs="*", help="image files or folders")
    parser.add_argument("-o", "--output", help="output file, or folder for several inputs")
    parser.add_argument("-e", "--eyes", default=None, metavar="MM",
                        help="distance between the two eyes, in millimetres, or auto to size it "
                             "to the scene. 65 is the human average, and worth trying, but a "
                             "close-up wants less and a landscape a great deal more")
    parser.add_argument("-f", "--focus", default=None, metavar="METRES",
                        help="how far away the screen plane sits, or auto. Whatever is at this "
                             "distance has no separation; nearer comes forward, further recedes")
    parser.add_argument("-t", "--target", type=float, default=None, metavar="PCT",
                        help="what auto aims for: near-to-far separation as a %% of frame width. "
                             "of frame width")
    parser.add_argument("--limit", type=float, default=3.0, metavar="PCT",
                        help="ceiling on separation as a %% of frame width, so something very "
                             "close cannot demand more parallax than an eye can fuse")
    parser.add_argument("-m", "--model", choices=("da3", "da2-small", "da2-base", "da2-large"),
                        default="da3",
                        help="da3 measures depth in metres; the da2 models only rank it, and are "
                             "fitted onto an assumed range")
    parser.add_argument("--depth-size", default=None,
                        help="longest side fed to the depth model (shortest, for the da2 models), "
                             "or auto to follow the photo")
    # The old names meant something this no longer computes.  Failing loudly beats
    # translating them into a number that only looks like what was asked for.
    parser.add_argument("-d", "--disparity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("-c", "--convergence", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cross", action="store_true", help="write right|left for cross-eyed viewing")
    parser.add_argument("--max-size", type=int, default=0, help="cap the output width, 0 for native")
    parser.add_argument("--format", choices=("auto", "jpg", "png"), default="auto", dest="fmt")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--save-depth", action="store_true", help="also write a 16-bit depth map")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--oversize", choices=("ask", "skip", "resize"), default="ask",
                        help="a photo too big for memory: ask, skip it, or resize it to fit")

    parser.add_argument("--gui", action="store_true", help="open the desktop window instead")
    parser.add_argument("-V", "--version", action="version", version=f"StereoCraft {__version__}")
    return parser


def _number(value):
    """A setting that is either a measurement or the word auto."""
    if value is None or str(value).lower() == "auto":
        return "auto"
    return float(value)


def settings_for(args):
    """The settings a run converts with."""
    common = dict(
        model=args.model,
        focus_m=_number(args.focus),
        limit_pct=args.limit,
        cross_eyed=args.cross,
        device=args.device,
        on_oversize=oversize_handler(args.oversize),
    )
    settings = Settings(**common, max_size=args.max_size, quality=args.quality,
                        fmt=args.fmt, save_depth=args.save_depth)
    if args.eyes is not None:
        settings.eyes_mm = _number(args.eyes)
    if args.target is not None:
        settings.target_pct = args.target
    if args.depth_size is not None:
        settings.depth_size = args.depth_size
    return settings


def retired(args):
    """The settings that used to exist, and what to say to someone still using them.

    Depth is measured in metres now, so a percentage of frame width and a
    normalised screen plane no longer describe anything the renderer does.  A
    plausible-looking translation would quietly convert to a different picture
    than the one that was asked for, which is worse than stopping.
    """
    for old, new, why in (
        ("disparity", "--eyes MM",
         "separation is worked out from the eye distance and the scene, not set as a percentage"),
        ("convergence", "--focus METRES",
         "the screen plane is a real distance now, not a position in a normalised range"),
    ):
        if getattr(args, old) is not None:
            return f"--{old} is gone: use {new} instead, because {why}."
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    retirement = retired(args)
    if retirement:
        print(retirement, file=sys.stderr)
        return 2
    if args.gui:
        from .gui import main as gui_main

        return gui_main()

    if not args.inputs:
        build_parser().print_help()
        return 1
    photos = collect(args.inputs)
    if not photos:
        print("nothing to convert", file=sys.stderr)
        return 1
    if args.output and len(photos) > 1 and Path(args.output).suffix:
        print("with several inputs, --output must be a folder", file=sys.stderr)
        return 2

    converter = Converter(settings_for(args))
    failures = skipped = 0
    for index, item in enumerate(photos, 1):
        prefix = f"[{index}/{len(photos)}] " if len(photos) > 1 else ""
        try:
            info = converter.convert(item, args.output)
        except Exception as error:  # keep a batch going when one photo is broken
            failures += 1
            print(f"{prefix}{item.name}: {error}", file=sys.stderr)
            continue
        if info is None:  # too big, and the answer was to skip it
            skipped += 1
            print(f"{prefix}{item.name}: skipped", file=sys.stderr)
            continue
        width, height = info["output_size"]
        note = ""
        if info["resized_from"]:
            was, now = info["resized_from"], info["source_size"]
            note = f"  (resized from {was[0]}x{was[1]} to {now[0]}x{now[1]})"
        print(f"{prefix}{info['output']}  {width}x{height}  {info['seconds']:.1f}s{note}")
    if skipped:
        print(f"{skipped} photo{'s' if skipped > 1 else ''} skipped as too large", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
