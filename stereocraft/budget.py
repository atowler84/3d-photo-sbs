"""How large a photo this machine can take, and what to shrink it to when it cannot.

Nothing here is tuned to a particular computer.  Both halves of the estimate are
properties of the code and the network rather than of any one machine:

The **full-resolution stage** is a fixed set of float32 copies of the frame.  At
its worst moment, while the finished pair is being turned into bytes, the photo
(3 channels) and its grey guide, the depth map, both rendered eyes, the composed
pair (6) and the two temporaries clamping and rounding it (6 each) are all live
at once -- about 122 bytes for every pixel of the source photo.  Below roughly
10 MP the banded renderer is the high-water mark instead, and comes out close
enough to the same figure that one number covers both.

The **depth network** is sized by how many activation elements it keeps alive
per pixel it is fed, which is a fact about the architecture, not the hardware:
measured within a few percent of each other on CUDA and CPU, at fp16 and fp32,
across all three model sizes.  Multiplying by the working resolution and the
dtype in use lets a small model on a modest card be judged on what it actually
costs, instead of on what the large one would.

Both are rounded up.  Guessing high costs a few percent of resolution; guessing
low costs the conversion.
"""

import ctypes
import os
import sys

import torch

from .depth import dtype_for

BYTES_PER_PIXEL = 128
# Activation elements the depth network holds per pixel of its own input.
ACTIVATIONS = {"small": 160, "base": 290, "large": 480}
# What is left over absorbs fragmentation, the image encoder, and whatever else
# the machine is busy with.
HEADROOM = 0.85


def _available_ram():
    """Bytes of system memory that can be handed out right now, or None."""
    try:  # Linux: MemAvailable is the kernel's own answer to this question
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys
        return None
    try:  # macOS and the BSDs; free pages only, so an honest underestimate
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def free_bytes(device):
    """Memory available to a conversion right now, or None if we cannot tell.

    The model is already resident by the time this is asked, so what comes back
    excludes it on every path.
    """
    if device.type == "cuda":
        # Torch's caching allocator holds on to blocks between conversions, and
        # the driver counts those as taken.  They are free to the next photo,
        # so a batch must add them back or the second photo onwards looks much
        # poorer than it is and falls back to the CPU for no reason.
        spare = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return torch.cuda.mem_get_info(device)[0] + spare
    if device.type == "mps":
        # Apple's unified memory: the driver publishes a ceiling for Torch, and
        # what it has already taken counts against it.
        try:
            return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            return _available_ram()
    if device.type == "cpu":
        return _available_ram()
    return None  # an accelerator we know nothing about: let it find out by trying


def depth_cost(estimator, width, height, size="auto", device=None):
    """Bytes the depth network wants for a photo of this shape.

    `device` prices somewhere the model is not currently loaded, which is how
    the CPU can be asked what the GPU would make of a photo and the other way
    round.  Only the dtype differs, and the working resolution is arithmetic on
    the photo's shape, so neither needs the weights to be there.
    """
    working = estimator.working_size(height, width, size)
    per_pixel = ACTIVATIONS.get(estimator.name, max(ACTIVATIONS.values()))
    return per_pixel * working[0] * working[1] * dtype_for(device or estimator.device).itemsize


def needs(estimator, width, height, size="auto", device=None):
    """Bytes a conversion of this photo is expected to want at its peak.

    The two stages run one after the other and the first has let go before the
    second starts, so the larger of the two is what has to fit.
    """
    return max(depth_cost(estimator, width, height, size, device), width * height * BYTES_PER_PIXEL)


def fits(estimator, width, height, size="auto", device=None):
    """Is there room for this photo?  True when we cannot tell, so an unfamiliar
    machine behaves as it always did and finds out by trying."""
    device = device or estimator.device
    free = free_bytes(device)
    return True if free is None else needs(estimator, width, height, size, device) <= free * HEADROOM


def smaller_model(estimator, width, height, size="auto", device=None):
    """The largest of the smaller models that would leave room here, or None.

    Worth asking when nothing fits, because the depth network's working set is
    the floor under every conversion and a lighter model lowers it -- which is
    the difference between "free some memory" and something to actually do.
    """
    device = device or estimator.device
    free = free_bytes(device)
    if free is None:
        return None
    room = free * HEADROOM
    working = estimator.working_size(height, width, size)
    cost = working[0] * working[1] * dtype_for(device).itemsize
    order = [name for name in ("large", "base", "small") if name in ACTIVATIONS]
    lighter = order[order.index(estimator.name) + 1:] if estimator.name in order else order
    return next((name for name in lighter if ACTIVATIONS[name] * cost < room), None)


def plan(estimator, width, height, size="auto", device=None):
    """Largest size with this photo's shape that is expected to fit.

    Returns `(width, height)`, or None when no resize would help -- the depth
    network's own working set does not shrink with the photo, so once that alone
    is more than the machine has left, there is nothing to offer.
    """
    device = device or estimator.device
    free = free_bytes(device)
    if free is None:
        return None
    budget = free * HEADROOM
    if budget <= depth_cost(estimator, width, height, size, device):
        return None
    scale = min(1.0, (budget / BYTES_PER_PIXEL) / (width * height)) ** 0.5
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return None if target == (width, height) else target
