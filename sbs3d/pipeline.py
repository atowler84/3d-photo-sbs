"""The whole pipeline: photo in, side-by-side 3D photo out."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pillow_heif  # teaches Pillow to read the HEIC files iPhones produce
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import budget, stereo
from .depth import DepthEstimator

pillow_heif.register_heif_opener()

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class Settings:
    model: str = "large"
    depth_size: object = "auto"
    disparity: float = 2.0
    convergence: float = 0.9
    cross_eyed: bool = False
    max_size: int = 0
    quality: int = 95
    fmt: str = "auto"
    save_depth: bool = False
    device: str = "auto"
    # Called when a photo will not fit, with the `TooBig` describing it, and
    # expected to return "resize" or "skip".  Left unset nothing is ever
    # silently downscaled: the photo is skipped and the caller told why.
    on_oversize: object = None


class TooBig(Exception):
    """A photo that will not fit in memory, and the resize that would fit.

    `target` is the size to convert at instead, or None when no size would help.
    """

    def __init__(self, path, source, target, device, wanted, free, disparity=Settings.disparity,
                 alternative=None):
        self.path = Path(path)
        self.source = source
        self.target = target
        self.device = device
        self.wanted = wanted
        self.free = free
        self.disparity = disparity
        self.alternative = alternative
        super().__init__(self.describe())

    def _sbs_width(self, width):
        """Width of the finished pair: two eyes, less the margin `make_pair`
        trims off each end where one eye has no pixels to show."""
        return 2 * (width - 2 * round(width * self.disparity / 200))

    @staticmethod
    def _mp(size):
        return f"{size[0]}x{size[1]} ({size[0] * size[1] / 1e6:.1f} MP)"

    def describe(self):
        where = "video memory" if self.device.type == "cuda" else "memory"
        lines = [
            f"{self.path.name} is {self._mp(self.source)} and needs about "
            f"{self.wanted / 1e9:.1f} GB, but only {self.free / 1e9:.1f} GB of "
            f"{where} is free."
        ]
        if self.target is None:
            lines.append("No smaller size would fit either: the depth model needs that much"
                         " whatever size the photo is.")
            lines.append(f"The {self.alternative} depth model would fit, and costs little in"
                         f" quality (--model {self.alternative})." if self.alternative
                         else "Free some memory and try again.")
        else:
            keep = 100 * (self.target[0] * self.target[1]) / (self.source[0] * self.source[1])
            lines.append(
                f"Resizing to {self._mp(self.target)} would fit -- "
                f"{self.target[0] / self.source[0]:.0%} of the width, {keep:.0f}% of the pixels."
            )
            lines.append(
                f"The side-by-side image would come out about "
                f"{self._sbs_width(self.target[0])}x{self.target[1]} instead of "
                f"{self._sbs_width(self.source[0])}x{self.source[1]}."
            )
        return "\n".join(lines)


def photo_size(path):
    """The photo's dimensions without decoding it, EXIF orientation included."""
    with Image.open(path) as img:
        width, height = img.size
        if img.getexif().get(274, 1) in (5, 6, 7, 8):  # 274 is Orientation
            width, height = height, width
        return width, height


def load_image(path, size=None):
    """Read a photo, honouring the EXIF orientation cameras write.

    `size` resizes on the way in.  JPEGs take the shortcut of decoding straight
    to a smaller raster, so an oversized photo never has to exist full size.
    """
    with Image.open(path) as img:
        if size is not None:
            # draft() sees the raster as stored, so a photo the EXIF turns on
            # its side wants the target the same way round.
            sideways = img.getexif().get(274, 1) in (5, 6, 7, 8)
            img.draft("RGB", size[::-1] if sideways else size)  # JPEG only; a no-op elsewhere
            img = ImageOps.exif_transpose(img).convert("RGB")
            return np.array(img.resize(size, Image.LANCZOS) if img.size != size else img)
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def output_path(src, dst, fmt):
    src = Path(src)
    ext = src.suffix.lower() if fmt == "auto" else f".{fmt}"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    if dst is None:
        return src.with_name(f"{src.stem}_sbs{ext}")
    dst = Path(dst)
    if dst.is_dir() or dst.suffix == "":
        return dst / f"{src.stem}_sbs{ext}"
    return dst


def save_image(array, path, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(array)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(path, quality=quality, subsampling=0, optimize=True)
    elif suffix == ".png":
        img.save(path, compress_level=6)
    else:
        img.save(path)
    return path


def _area(size):
    """Pixels in a proposed size, and less than nothing for "no size at all",
    so that any real offer beats having none."""
    return size[0] * size[1] if size else -1


# Every way a photo can fail to fit, so that one `except` covers the lot.
OUT_OF_MEMORY = (torch.cuda.OutOfMemoryError, MemoryError, TooBig)


def _to_uint8(tensor):
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


class Converter:
    """Holds the depth model between conversions so batches pay for it once."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._depth = None

    @property
    def depth_model(self):
        if self._depth is None or self._depth.name != self.settings.model:
            self._depth = DepthEstimator(self.settings.model, self.settings.device)
        return self._depth

    def convert(self, src, dst=None):
        """Convert one photo.

        A photo that will not fit in video memory is retried on the CPU.  One
        that will not fit there either is put to `Settings.on_oversize`, which
        answers "resize" to convert it smaller or "skip" to leave it alone;
        skipping returns None rather than raising, since it is a choice and not
        a failure.
        """
        try:
            return self._attempt(src, dst)
        except TooBig as oversize:
            # The handler hears about it either way, including the hopeless
            # case, so that a skip is never a silent one.
            if self._decide(oversize) != "resize" or oversize.target is None:
                return None
            return self._attempt(src, dst, oversize.target)

    def _decide(self, oversize):
        ask = self.settings.on_oversize
        # Without someone to ask, nothing is quietly downscaled behind the
        # caller's back -- the photo is skipped and they are told about it.
        if ask is None:
            print(oversize.describe(), file=sys.stderr)
            return "skip"
        return ask(oversize)

    def _attempt(self, src, dst, size=None):
        """Run once, falling back from the GPU to the CPU if room runs out.

        Every way of running out -- the pre-flight check, CUDA's own error, a
        failed host allocation -- leaves by the same door, as a `TooBig` for
        whichever device was the last to turn the photo away.
        """
        try:
            return self._run(src, dst, size)
        except OUT_OF_MEMORY as problem:
            gpu = self.depth_model.device
            if gpu.type == "cpu":
                raise self._too_big(problem, src, size, [gpu]) from None
            print(f"{Path(src).name}: too big for GPU memory, trying the CPU", file=sys.stderr)
            requested, self.settings.device, self._depth = self.settings.device, "cpu", None
            torch.cuda.empty_cache()
            try:
                return self._run(src, dst, size)
            except OUT_OF_MEMORY as fallback_problem:
                oversize = self._too_big(fallback_problem, src, size, [gpu, self.depth_model.device])
            finally:  # leave the next photo free to try the GPU again
                self.settings.device, self._depth = requested, None
            raise oversize from None

    def _too_big(self, problem, src, size, devices):
        """Turn a memory failure into a proposal, sized for what is free now."""
        source = problem.source if isinstance(problem, TooBig) else (size or photo_size(src))
        return self._proposal(src, source, devices)

    def _proposal(self, src, source, devices):
        """The best offer among the devices still in play.

        Priced against the model that is really loaded, so a small model on a
        modest card is judged by what it costs rather than by what the large one
        would.  Both devices are asked, because the roomier of the two is not
        always the GPU -- a generous card in a busy machine can have more to
        spare than the system does -- and the better offer is the one to make.
        """
        estimator, size = self.depth_model, self.settings.depth_size
        offers = [(budget.plan(estimator, *source, size, device), device) for device in devices]
        target, device = max(offers, key=lambda offer: _area(offer[0]))
        return TooBig(src, source, target, device,
                      budget.needs(estimator, *source, size, device),
                      budget.free_bytes(device) or 0, self.settings.disparity,
                      None if target else budget.smaller_model(estimator, *source, size, device))

    def _run(self, src, dst, size=None):
        cfg = self.settings
        started = time.perf_counter()
        estimator = self.depth_model
        device = estimator.device

        # Asking the file how big it is costs nothing, so a photo that cannot
        # fit is turned away before it is ever decoded.  That matters most on
        # the CPU path, where the alternative is the kernel's OOM killer taking
        # the process with it rather than an exception anyone can catch.
        if size is None:
            source = photo_size(src)
            if not budget.fits(estimator, *source, cfg.depth_size):
                raise self._proposal(src, source, [device])

        image = load_image(src, size)
        height, width = image.shape[:2]

        raw_depth = estimator(image, cfg.depth_size)

        rgb = torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1).float().div_(255.0)
        guide = rgb.mean(0)[None, None]
        depth = stereo.guided_upsample(stereo.normalize(raw_depth), guide)[0, 0].clamp_(0, 1)

        left, right = stereo.make_pair(rgb, depth, cfg.disparity, cfg.convergence)
        sbs = stereo.compose(left, right, cfg.cross_eyed)

        if cfg.max_size and sbs.shape[2] > cfg.max_size:
            scale = cfg.max_size / sbs.shape[2]
            size = (max(1, round(sbs.shape[1] * scale)), cfg.max_size)
            sbs = F.interpolate(sbs[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]

        out = save_image(_to_uint8(sbs), output_path(src, dst, cfg.fmt), cfg.quality)
        if cfg.save_depth:
            gray = (depth * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16)
            Image.fromarray(gray, mode="I;16").save(out.with_name(f"{Path(src).stem}_depth.png"))

        return {
            "input": Path(src),
            "output": out,
            "source_size": (width, height),
            "resized_from": photo_size(src) if size else None,
            "output_size": (sbs.shape[2], sbs.shape[1]),
            "seconds": time.perf_counter() - started,
        }


def convert(src, dst=None, **kwargs):
    """One-shot helper for scripts: `sbs3d.convert("photo.jpg", disparity=2.5)`."""
    return Converter(Settings(**kwargs)).convert(src, dst)
