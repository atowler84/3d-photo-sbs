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

from . import stereo
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


def load_image(path):
    """Read a photo, honouring the EXIF orientation cameras write."""
    with Image.open(path) as img:
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
        """Convert one photo, retrying on the CPU if the GPU runs out of room."""
        try:
            return self._run(src, dst)
        except torch.cuda.OutOfMemoryError:
            if self.depth_model.device.type == "cpu":
                raise
            print(f"{Path(src).name}: out of GPU memory, retrying on the CPU", file=sys.stderr)
            requested, self.settings.device, self._depth = self.settings.device, "cpu", None
            torch.cuda.empty_cache()
            try:
                return self._run(src, dst)
            finally:  # leave the next photo free to try the GPU again
                self.settings.device, self._depth = requested, None

    def _run(self, src, dst):
        cfg = self.settings
        started = time.perf_counter()
        estimator = self.depth_model
        device = estimator.device

        image = load_image(src)
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
            "output_size": (sbs.shape[2], sbs.shape[1]),
            "seconds": time.perf_counter() - started,
        }


def convert(src, dst=None, **kwargs):
    """One-shot helper for scripts: `sbs3d.convert("photo.jpg", disparity=2.5)`."""
    return Converter(Settings(**kwargs)).convert(src, dst)
