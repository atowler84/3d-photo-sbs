# -*- mode: python ; coding: utf-8 -*-
"""Portable Windows build: one folder, two exes, no installer.

Built by `build.ps1`, which stages a copy of the source next to this file and
drops the weights into the finished folder afterwards.
"""

import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir, os.pardir))
ICON = os.path.join(SPECPATH, "stereocraft.ico")

hiddenimports = [
    "stereocraft.cli",
    "stereocraft.gui",
    # transformers reaches its model classes by name through the auto mappings,
    # so there is no import statement for the analysis to follow to them.
    "transformers.models.depth_anything.configuration_depth_anything",
    "transformers.models.depth_anything.modeling_depth_anything",
    "transformers.models.dinov2.configuration_dinov2",  # Depth-Anything's backbone
    "transformers.models.dinov2.modeling_dinov2",
] + collect_submodules("transformers.models.auto")

# transformers decides at import time whether Torch is there by asking the
# installed metadata for its version, so the .dist-info has to come along or
# the whole library concludes it has no backend.
datas = copy_metadata("torch", recursive=True) + copy_metadata("transformers", recursive=True)

# HEIC support is a compiled library that nothing imports by name.
binaries = collect_dynamic_libs("pillow_heif")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # None of these are used, and each one that creeps in through an optional
    # import of transformers costs hundreds of megabytes.
    excludes=[
        "torchvision", "torchaudio", "tensorflow", "jax", "flax", "keras",
        "matplotlib", "scipy", "pandas", "IPython", "notebook", "pytest", "cv2",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="StereoCraft",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX on Torch's DLLs is slow to build and a good way to break them
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)

cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="StereoCraft-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=ICON,
)

coll = COLLECT(
    gui, cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="StereoCraft",
)
