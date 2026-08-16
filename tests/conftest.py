"""Fixtures for the test suite.

Everything the tests need is generated here rather than committed: a repository
of sample media rots, and a two-second clip built by ffmpeg is both smaller and
more honest about what it contains.

The tests that need the depth model are marked `slow`.  They are the ones worth
having -- most of what has broken here has broken end to end -- but they cost a
model load, so `-m "not slow"` gets the arithmetic in a second or two.
"""

import subprocess

import numpy as np
import pytest
from PIL import Image


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the depth model loaded")


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def photo(tmp_path_factory):
    """A small photo with something for the depth model to find: a bright block
    on a dark ground reads as a near object in front of a far one."""
    path = tmp_path_factory.mktemp("media") / "photo.jpg"
    a = np.zeros((240, 320, 3), np.uint8)
    a[:, :] = (30, 40, 60)
    a[80:200, 100:220] = (220, 200, 180)
    Image.fromarray(a).save(path, quality=95)
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """Two seconds of moving picture with a soundtrack, 60 frames at 30fps."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))
    return path


@pytest.fixture(scope="session")
def silent_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


@pytest.fixture(scope="session")
def converter():
    """One depth model for the whole session; loading it is the expensive part."""
    from stereocraft.pipeline import Converter, Settings
    c = Converter(Settings())
    c.depth_model
    return c


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().rstrip(","))


def frame_sizes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "frame=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip().rstrip(",") for line in out.splitlines() if line.strip()}


def audio_codec(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return out.strip() or None
