"""Monocular depth estimation with Depth-Anything V2."""

import os
import sys

import numpy as np
import torch

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

# Patch size of the ViT backbone; input dimensions must be a multiple of it.
PATCH = 14
# Depth-Anything is trained at a 518px short side.  Feeding it more resolves finer
# structure, but drifting too far from training makes the overall depth wobble, so
# "auto" follows the photo between the trained size and twice it.
MIN_SIZE, MAX_SIZE = 518, 1036

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _app_dir():
    """Where the app's own files sit: beside the exe once frozen, else the repo."""
    if getattr(sys, "frozen", False):  # the packaged Windows build
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _use_local_cache():
    """Prefer the checkpoints already sitting next to the app over ~/.cache."""
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return
    local = os.path.join(_app_dir(), "hf-cache")
    if os.path.isdir(local):
        os.environ["HF_HUB_CACHE"] = local


def checkpoint(model):
    """Where to load `model` from.

    Weights shipped with the app win.  The portable build puts them in a plain
    `models/large` folder beside the exe, and a folder is something
    `from_pretrained` reads without touching the network at all -- so it starts
    the same whether or not the machine it landed on has any internet.
    """
    bundled = os.path.join(_app_dir(), "models", model)
    if os.path.isdir(bundled):
        return bundled
    _use_local_cache()
    return MODELS[model]


def dtype_for(device):
    """fp16 halves the memory traffic on a GPU and is indistinguishable here;
    CPU stays fp32.  `budget` prices the other device with this too, so an
    estimate never disagrees with what would actually run."""
    return torch.float16 if device.type == "cuda" else torch.float32


def pick_device(request="auto"):
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthEstimator:
    """Loads one Depth-Anything V2 checkpoint and keeps it resident."""

    def __init__(self, model="large", device="auto"):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}, pick one of {sorted(MODELS)}")
        source = checkpoint(model)
        from transformers import AutoModelForDepthEstimation  # heavy; import on demand

        self.device = pick_device(device)
        self.name = model
        self.dtype = dtype_for(self.device)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(source, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        mean = torch.tensor(_MEAN, device=self.device).view(3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(3, 1, 1)
        self._mean, self._std = mean, std

    @staticmethod
    def resolve_size(size, short_side):
        """Turn a `--depth-size` request into an actual short-side length."""
        if size in (None, 0, "auto"):
            return int(min(MAX_SIZE, max(MIN_SIZE, short_side)))
        return max(PATCH, int(size))

    def working_size(self, height, width, size="auto"):
        """The resolution the network will actually run at for this shape.

        Shared with `budget`, which prices a conversion before one happens, so
        that what is charged for and what is run are never two different sizes.
        """
        scale = self.resolve_size(size, min(height, width)) / min(height, width)
        return (max(PATCH, int(round(height * scale / PATCH)) * PATCH),
                max(PATCH, int(round(width * scale / PATCH)) * PATCH))

    @torch.inference_mode()
    def __call__(self, image, size="auto"):
        """Estimate relative inverse depth (bigger = nearer).

        `image` is a uint8 HxWx3 array. Returns a float32 tensor at the network's
        own working resolution -- `stereo.guided_upsample` lifts it back to full
        resolution using the photo itself as a guide, which keeps depth edges
        glued to picture edges far better than plain interpolation would.
        """
        h, w = image.shape[:2]
        # Depth-Anything is trained with the *shorter* side at `size`, aspect kept.
        th, tw = self.working_size(h, w, size)

        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255.0)
        x = torch.nn.functional.interpolate(
            x[None], size=(th, tw), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0, 1)
        x = (x[0] - self._mean) / self._std

        depth = self.model(pixel_values=x[None].to(self.dtype)).predicted_depth
        return depth.reshape(1, 1, *depth.shape[-2:]).float()
