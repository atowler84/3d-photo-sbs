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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sbs3d",
        description="Turn a photo into a full-resolution side-by-side 3D image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="image files or folders")
    parser.add_argument("-o", "--output", help="output file, or folder for several inputs")
    parser.add_argument("-d", "--disparity", type=float, default=2.0,
                        help="eye separation, as a %% of image width; 1.5-3 is comfortable")
    parser.add_argument("-c", "--convergence", type=float, default=0.9,
                        help="depth that sits on the screen plane, 0 (far) to 1 (near)")
    parser.add_argument("-m", "--model", choices=("small", "base", "large"), default="large",
                        help="Depth-Anything V2 size")
    parser.add_argument("--depth-size", default="auto",
                        help="shorter side fed to the depth model, or auto to follow the photo")
    parser.add_argument("--cross", action="store_true", help="write right|left for cross-eyed viewing")
    parser.add_argument("--max-size", type=int, default=0, help="cap the output width, 0 for native")
    parser.add_argument("--format", choices=("auto", "jpg", "png"), default="auto", dest="fmt")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--save-depth", action="store_true", help="also write a 16-bit depth map")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--gui", action="store_true", help="open the desktop window instead")
    parser.add_argument("-V", "--version", action="version", version=f"sbs3d {__version__}")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
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

    settings = Settings(
        model=args.model,
        depth_size=args.depth_size,
        disparity=args.disparity,
        convergence=args.convergence,
        cross_eyed=args.cross,
        max_size=args.max_size,
        quality=args.quality,
        fmt=args.fmt,
        save_depth=args.save_depth,
        device=args.device,
    )
    converter = Converter(settings)
    failures = 0
    for index, photo in enumerate(photos, 1):
        prefix = f"[{index}/{len(photos)}] " if len(photos) > 1 else ""
        try:
            info = converter.convert(photo, args.output)
        except Exception as error:  # keep a batch going when one photo is broken
            failures += 1
            print(f"{prefix}{photo.name}: {error}", file=sys.stderr)
            continue
        width, height = info["output_size"]
        print(f"{prefix}{info['output']}  {width}x{height}  {info['seconds']:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
