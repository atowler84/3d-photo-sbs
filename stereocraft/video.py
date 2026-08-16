"""Side-by-side 3D video: the still pipeline, run over every frame of a clip.

Each frame gets exactly what a photo gets -- the same depth network, the same
guided upsample, the same splat-and-resample renderer.  Two things have to be
added around it, and both come from the picture moving rather than from anything
about video files:

**Depth has to hold still.**  Depth-Anything is a per-frame model, and a
per-frame estimate wobbles.  In a depth map that reads as noise; turned into a
stereo pair it is the *geometry* that wobbles, which is a great deal harder to
look at.  `TemporalDepth` is the answer, and what it smooths and in what order is
the whole of the difference between a clip that is comfortable and one that is
not.

**Frames have to stay the same size.**  `make_pair` trims the sliver at each
edge that only one eye can see, and sizes that trim from the frame it is given.
A still always uses the full depth range so that comes out constant; a smoothed
clip does not, so the trim is pinned for the whole clip up front instead.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import budget, stereo
from .depth import _app_dir
from .pipeline import OUT_OF_MEMORY, Converter, VideoSettings

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".wmv", ".flv"}

# Audio an mp4 will carry as it stands.  Anything else is re-encoded, which is a
# generation of loss on the soundtrack but beats refusing the file.
COPYABLE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}
# What to encode with, best first, per codec.  Which of these exists depends on
# how the ffmpeg to hand was built: an LGPL build carries no x264 or x265, both
# being GPL, so a portable app that ships its own cannot assume them.  Each entry
# is (encoder, quality flag, extra arguments) -- they do not agree on how quality
# is expressed, which is the other reason this cannot be one hardcoded name.
ENCODERS = {
    "h264": [("libx264", "-crf", ["-preset", "medium"]),
             ("h264_nvenc", "-cq", ["-rc", "vbr", "-preset", "p6"]),
             ("h264_qsv", "-global_quality", []),
             ("h264_amf", "-qp_i", []),
             ("libopenh264", "-q", []),
             ("h264_mf", "-q", [])],
    "hevc": [("libx265", "-crf", ["-preset", "medium"]),
             ("hevc_nvenc", "-cq", ["-rc", "vbr", "-preset", "p6"]),
             ("hevc_qsv", "-global_quality", []),
             ("hevc_amf", "-qp_i", []),
             ("hevc_mf", "-q", [])],
}
# Frame-to-frame change in depth, relative to the depth itself, that means the
# picture cut rather than moved.  Relative because metres have no fixed scale --
# an ordinary pan sits far below this at any distance.
SCENE_CUT = 0.35


class MissingFFmpeg(Exception):
    """ffmpeg is not installed, and nothing can be done about a video without it."""

    def __init__(self, name):
        super().__init__(
            f"{name} was not found. Video needs ffmpeg installed and on the PATH.\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Windows:       winget install ffmpeg\n"
            f"A copy of {name} sitting beside the app is used too, if there is one."
        )


def available_encoders():
    """The encoder names this ffmpeg was built with, asked once and remembered."""
    global _ENCODERS_SEEN
    if _ENCODERS_SEEN is None:
        result = subprocess.run([_tool("ffmpeg"), "-hide_banner", "-encoders"],
                                capture_output=True, text=True)
        _ENCODERS_SEEN = {line.split()[1] for line in result.stdout.splitlines()
                          if line.startswith(" ") and len(line.split()) > 1}
    return _ENCODERS_SEEN


def pick_encoder(codec):
    """The best encoder available for this codec, and how to ask it for quality.

    x264 first where it exists, being the best of them at a given size.  A build
    without it -- an LGPL one, most likely -- falls to the graphics card, and
    failing that to a software encoder that is not GPL.  Something is always
    there, so the choice never has to be explained to whoever is converting.
    """
    have = available_encoders()
    for name, quality, extra in ENCODERS.get(codec, ENCODERS["h264"]):
        if name in have:
            return name, quality, extra
    raise MissingFFmpeg(f"an encoder for {codec}")


_ENCODERS_SEEN = None


def _tool(name):
    """Find ffmpeg or ffprobe: on the PATH, or shipped next to the app."""
    if sys.platform == "win32":
        name += ".exe"
    found = shutil.which(name)
    if found:
        return found
    beside = os.path.join(_app_dir(), name)
    if os.path.exists(beside):
        return beside
    raise MissingFFmpeg(name)


@dataclass
class Clip:
    """What a video is, as far as any of this needs to care."""

    width: int
    height: int
    fps: float
    frames: int
    duration: float
    audio: object = None  # the audio stream's codec, or None if it is silent


def _fraction(text):
    """ffprobe writes frame rates as "30000/1001"; None for the 0/0 it uses to
    mean it does not know."""
    try:
        top, _, bottom = str(text).partition("/")
        rate = float(top) / float(bottom or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _rotation(stream):
    """How far the player is meant to turn this stream before showing it.

    Phones record sideways and note the rotation rather than rotating the
    pixels, so this is the difference between a portrait clip being understood
    as portrait and being handed to the depth model on its side.
    """
    tag = stream.get("tags", {}).get("rotate")
    if tag is not None:
        return int(round(float(tag)))
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"])))
    return 0


def probe(path):
    """Everything about a clip that the conversion needs, in one ffprobe call."""
    result = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{Path(path).name} could not be read: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{Path(path).name} has no video track in it")

    width, height = int(video["width"]), int(video["height"])
    if _rotation(video) % 180 == 90:  # -90 lands on 90 here too, which is the point
        width, height = height, width
    # The average rate rather than the nominal one: a variable-rate phone clip
    # quotes something optimistic as `r_frame_rate`, and the average is what
    # keeps the soundtrack level with the picture.
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")) or 30.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    audio = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
    return Clip(width, height, fps, frames, duration, audio)


class TemporalDepth:
    """The inverse depth map, with a memory of the frames before this one.

    This used to have two halves.  The larger one smoothed the percentile range
    that mapped raw depth onto 0-1, because measured afresh every frame it
    twitched as the scene moved -- a subject walking toward the camera shifted it
    and the whole map slid to compensate -- and that was the single biggest
    source of the wobble a per-frame model gave a clip.

    Metric depth removes the need for that range -- metres are metres whatever
    else is in the frame -- but it is worth being honest that this is not a free
    win.  Renormalising every frame was also cancelling the model's own global
    scale wobble, and measured against Depth-Anything V2 on a static shot, metric
    depth is about a third *noisier* frame to frame, not quieter.  Smoothing the
    metric scale back out was tried and made it worse: the noise is spread
    through the map rather than sitting in one global factor.

    It does not matter in the end, which is why a plain exponential average is
    all that is left here.  In the units that count -- how far the disparity
    field actually moves between frames -- both models sit around a tenth of a
    pixel before any smoothing at all, against roughly a third of a pixel for the
    smallest movement an eye can pick out.  The default halves that again.

    A cut is the one thing an average cannot be asked to sit through, so a frame
    that differs wholesale from the last is taken on its own and the memory
    starts again from there.  The test is relative, because depth in metres has
    no fixed scale to set an absolute threshold against: a room and a landscape
    are both ordinary, and differ by two orders of magnitude.
    """

    def __init__(self, keep=0.5, cut=SCENE_CUT):
        # 1.0 would freeze the first frame's depth over the whole clip, so the
        # knob stops short of it however far it is turned up.
        self.keep = min(max(float(keep), 0.0), 0.95)
        self.cut = cut
        self.previous = None
        self.cuts = 0

    def reset(self):
        self.previous = None

    def __call__(self, inverse):
        if self.previous is not None and self.previous.shape == inverse.shape:
            scale = float(inverse.abs().mean()) + 1e-9
            if float((inverse - self.previous).abs().mean()) / scale > self.cut:
                self.cuts += 1
                self.reset()

        if self.previous is None:  # the first frame, or the first after a cut
            self.previous = inverse
            return inverse

        depth = torch.lerp(inverse, self.previous, self.keep)
        self.previous = depth
        return depth


def clock(seconds):
    """A duration at the coarseness someone waiting on it actually reads."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _even(value):
    """yuv420p halves both dimensions, so both have to be even to start with."""
    return max(2, int(value) - int(value) % 2)


@dataclass
class Geometry:
    """Where every frame of this clip ends up, worked out once for all of them."""

    margin: int  # trimmed off each end of each eye, pinned so no frame differs
    eye: int  # width of one eye view in the finished frame
    height: int

    @property
    def width(self):
        return 2 * self.eye


def geometry(clip, settings):
    """The finished frame's shape.

    Half width per eye by default, which puts the clip out at the size it came
    in.  That is what players and headsets expect and what their hardware
    decoders can keep up with; `full_width` keeps every native pixel instead and
    doubles the frame, which is past what most of them will decode above 1080p.
    """
    margin = stereo.max_margin(clip.width, settings.limit_pct)
    eye = clip.width - 2 * margin if settings.full_width else clip.width // 2
    return Geometry(margin=margin, eye=_even(eye), height=_even(clip.height))


def output_path(src, dst=None):
    src = Path(src)
    if dst is None:
        return src.with_name(f"{src.stem}_sbs.mp4")
    dst = Path(dst)
    if dst.is_dir() or dst.suffix == "":
        return dst / f"{src.stem}_sbs.mp4"
    return dst


def _read_exactly(stream, view):
    """Fill `view` from `stream`, or report how far it got at the end of the file.

    A pipe hands over whatever it has rather than what was asked for, so a frame
    arrives in as many pieces as the operating system feels like.
    """
    got = 0
    while got < len(view):
        read = stream.readinto(view[got:])
        if not read:
            break
        got += read
    return got


def _decoder(src, clip, size, stderr):
    """ffmpeg reading the clip out as raw RGB, one frame after another.

    Constant frame rate on the way out even if the file is variable: the raw
    frames carry no timestamps, so anything else would leave the picture and the
    soundtrack drifting apart over the length of the clip.  Rotation is applied
    by ffmpeg itself, so frames arrive the right way up.
    """
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(src)]
    if size is not None:
        args += ["-vf", f"scale={size[0]}:{size[1]}"]
    args += ["-fps_mode", "cfr", "-r", f"{clip.fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)


def _encoder(out, src, clip, geo, cfg, stderr):
    """ffmpeg taking finished frames on stdin, with the original soundtrack
    alongside them straight off the source file."""
    args = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{geo.width}x{geo.height}", "-r", f"{clip.fps}", "-i", "-"]
    sound = bool(cfg.audio and clip.audio)
    if sound:
        # No -shortest here.  It looks like the safe option and is not: an AAC
        # track carries a little encoder priming, which makes ffmpeg reckon it
        # the shorter stream and truncate the picture to match.  On a 90-frame
        # clip that quietly cost three frames off the end.  Every frame rendered
        # is a frame worth keeping; a fractionally long soundtrack is harmless.
        args += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0"]
    encoder, quality, extra = pick_encoder(cfg.codec)
    if encoder != ENCODERS[cfg.codec][0][0]:
        # Worth a word.  The preferred encoder is the best of them at a given
        # size, and falling past it is invisible in the finished file unless
        # someone thinks to ask ffprobe -- so a clip that came out softer than
        # the last one would look like the app's fault rather than the ffmpeg's.
        print(f"{Path(out).name}: encoding with {encoder}; this ffmpeg has no "
              f"{ENCODERS[cfg.codec][0][0]}", file=sys.stderr)
    args += ["-c:v", encoder, quality, str(cfg.crf), *extra,
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if sound:
        # Copied when the container will take it as it is, so the soundtrack goes
        # through untouched; re-encoded only when it would otherwise be refused.
        args += (["-c:a", "copy"] if clip.audio in COPYABLE_AUDIO
                 else ["-c:a", "aac", "-b:a", "192k"])
    args.append(str(out))
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=stderr, bufsize=0)


def _decode_size(converter, src, clip):
    """The size to decode at: native, something smaller that fits, or nothing.

    One question for the whole file rather than one per frame, since every frame
    of a clip is the same size -- and asked before a single frame is decoded.
    Returns None for native, a `(width, height)` to scale to on the way in, or
    False when the answer was to skip the clip.
    """
    cfg = converter.settings
    estimator = converter.depth_model
    if budget.fits(estimator, clip.width, clip.height, cfg.depth_size):
        return None
    oversize = converter._proposal(src, (clip.width, clip.height), [estimator.device])
    if converter._decide(oversize) != "resize" or oversize.target is None:
        return False
    return _even(oversize.target[0]), _even(oversize.target[1])


def _fall_back_to_cpu(converter):
    """Move off the GPU mid-clip, for when something else on the machine takes
    the video memory out from under a conversion already forty frames in.

    Worth doing rather than giving up: the frames already encoded stay encoded
    and the clip finishes slowly, instead of the work being thrown away.
    """
    if converter.depth_model.device.type == "cpu":
        return False
    print("out of video memory part-way through; carrying on with the CPU", file=sys.stderr)
    converter.settings.device = "cpu"
    converter._depth = None
    torch.cuda.empty_cache()
    return True


def _squeeze(eye, width, height):
    """Resize one eye view to its place in the finished frame.

    Each eye separately, never the finished pair: resampling across the seam
    would blend the two eyes into one another at the join.
    """
    if eye.shape[-2:] == (height, width):
        return eye
    return F.interpolate(eye[None], size=(height, width), mode="bilinear",
                         align_corners=False, antialias=True)[0]


def convert_video(src, dst=None, converter=None, on_progress=None, on_frame=None, **kwargs):
    """Convert one clip into a side-by-side 3D one.

    `converter` keeps the depth model loaded across several clips; without one,
    `kwargs` are `VideoSettings` fields for a single conversion.  `on_progress`
    is called with `(frames done, frames expected, seconds so far)` and returning
    False from it cancels, leaving no half-written file behind.  `on_frame` is
    handed each finished frame as it goes by, for showing the work in progress;
    it costs nothing, the array having had to be made for the encoder anyway.

    Returns what was made, or None if it was cancelled or skipped.
    """
    if converter is None:
        converter = Converter(VideoSettings(**kwargs))
    elif kwargs:
        raise TypeError("give convert_video a converter or settings, not both")

    src = Path(src)
    cfg = converter.settings
    started = time.perf_counter()
    clip = probe(src)
    source = (clip.width, clip.height)

    size = _decode_size(converter, src, clip)
    if size is False:  # too big, and the answer was to leave it alone
        return None
    if size is not None:
        clip = Clip(size[0], size[1], clip.fps, clip.frames, clip.duration, clip.audio)

    if converter.depth_model.device.type == "cpu":
        # Roughly five and a half seconds per megapixel of network input, measured
        # on an eight-core desktop.  Rough is enough: the point is to say "hours"
        # before someone finds out by waiting, not to be right to the minute.
        work = converter.depth_model.working_size(clip.height, clip.width, cfg.depth_size)
        each = 5.4 * work[0] * work[1] / 1e6
        total = each * (clip.frames or 1)
        if total > 600:
            print(f"{src.name}: this is a CPU conversion -- about {each:.0f}s a frame, so "
                  f"{clock(total)} for {clip.frames} frames. A graphics card does it in "
                  f"minutes, and --model da2-small is the quickest way through without one.",
                  file=sys.stderr)

    geo = geometry(clip, cfg)
    out = output_path(src, dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalizer = TemporalDepth(cfg.temporal)

    frame_bytes = clip.width * clip.height * 3
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    # Reused rather than reallocated per frame, and writable so that handing it
    # to Torch does not have to copy it first.
    frame = np.frombuffer(buffer, np.uint8).reshape(clip.height, clip.width, 3)

    done, cancelled = 0, False
    # What the caller asked for, to be put back afterwards: a clip that had
    # to finish on the CPU should not decide that for the clips behind it,
    # any more than one oversized photo decides it for the rest of a folder.
    requested = cfg.device
    try:
        with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
            decoder = _decoder(src, clip, size, decode_log)
            encoder = _encoder(out, src, clip, geo, cfg, encode_log)
            try:
                while _read_exactly(decoder.stdout, view) == frame_bytes:
                    try:
                        left, right, _ = converter.render(frame, normalizer, geo.margin)
                    except OUT_OF_MEMORY:
                        if not _fall_back_to_cpu(converter):
                            raise
                        normalizer.reset()  # its memory is on a device we have just left
                        left, right, _ = converter.render(frame, normalizer, geo.margin)

                    left = _squeeze(left, geo.eye, geo.height)
                    right = _squeeze(right, geo.eye, geo.height)
                    sbs = stereo.compose(left, right, cfg.cross_eyed)
                    pixels = ((sbs.clamp(0, 1) * 255).round().to(torch.uint8)
                              .permute(1, 2, 0).contiguous().cpu().numpy())
                    encoder.stdin.write(pixels.tobytes())
                    if on_frame is not None:
                        on_frame(pixels)

                    done += 1
                    if on_progress and on_progress(done, clip.frames, time.perf_counter() - started) is False:
                        cancelled = True
                        break
            except BrokenPipeError:
                # The encoder died, and its own complaint is more use than the broken
                # pipe that is all this end saw of it.
                _stop(decoder, encoder)
                out.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg could not encode {out.name}: {_log(encode_log)}") from None
            except BaseException:  # Ctrl-C, out of memory, a frame that would not render
                _stop(decoder, encoder)
                out.unlink(missing_ok=True)
                raise

            if cancelled:
                _stop(decoder, encoder)
                out.unlink(missing_ok=True)
                return None

            decoder.stdout.close()
            encoder.stdin.close()
            decoder.wait()
            encoder.wait()
            if not done:
                out.unlink(missing_ok=True)
                raise RuntimeError(f"no frames came out of {src.name}: {_log(decode_log)}")
            if encoder.returncode:
                raise RuntimeError(f"ffmpeg could not write {out.name}: {_log(encode_log)}")

        seconds = time.perf_counter() - started
        return {
            "input": src,
            "output": out,
            "source_size": (clip.width, clip.height),
            "resized_from": source if size else None,
            "output_size": (geo.width, geo.height),
            "frames": done,
            "cuts": normalizer.cuts,
            "fps": clip.fps,
            "seconds": seconds,
        }
    finally:
        if converter.settings.device != requested:
            converter.settings.device, converter._depth = requested, None


def _stop(*processes):
    for process in processes:
        for pipe in (process.stdin, process.stdout):
            if pipe and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _log(handle):
    """The last thing ffmpeg said, which is the part worth repeating."""
    handle.seek(0)
    text = handle.read().decode("utf-8", "replace").strip()
    return text.splitlines()[-1] if text else "no reason given"
