# StereoCraft

Turn a single photo into a **full-resolution side-by-side 3D image**.

One job, done well: estimate a depth map, re-project the picture into a left and
a right eye view, and write the pair as one frame you can drop straight into a
headset or view free-eyed. Nothing is downscaled along the way.

| Photo | Output | Time |
| --- | --- | --- |
| 1.9 MP | 3152 × 1197 | 0.5 s |
| 7.1 MP | 6040 × 2304 | 1.2 s |
| 12.5 MP | 6044 × 4080 | 1.4 s |

Measured on an RTX 4080 Super with the model already loaded; add about fifteen
seconds for the first photo of a session. The GPU path falls back to the CPU
automatically if a photo will not fit in video memory, and offers to resize a
photo that will not fit there either -- see [when a photo is too
big](#when-a-photo-is-too-big).

There is a CPU path too and it is a great deal slower, because most of the cost
is the depth network rather than the pixels: a snapshot costs nearly as much as
a raw. On a machine without a GPU the lighter `--model da2-large` is the trade
worth making, at some cost in geometry -- see [which
model](#which-model).

```
photo.jpg  ->  photo_sbs.jpg        (left | right, full width, no downscaling)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . --no-deps && pip install -r requirements.txt
```

The two steps are deliberate. Depth Anything 3's own dependency list replaces
Torch with a different CUDA build and pulls in a reconstruction and visualisation
stack — open3d, pycolmap, moviepy, flask, jupyter — that depth inference never
touches. `requirements.txt` carries what it actually needs at runtime, which was
worked out by blocking the rest and checking a photo still converted.

The desktop window additionally needs Tkinter, which some Python builds omit:

```bash
sudo apt install python3-tk
```

JPEG, PNG, HEIC, WebP, TIFF and BMP all work, so photos come straight off a
phone without converting anything first.

Depth weights download themselves on first run (~1.3 GB for the large model) and
are cached. If a `hf-cache/` folder exists next to the app it is used instead of
`~/.cache/huggingface`, so an existing download is picked up automatically.

## A Windows app

`packaging/windows/build.ps1` freezes the lot -- Python, Torch, the weights --
into one folder that runs on a Windows machine with nothing installed on it.
Build it on Windows, with any Python from 3.10 to 3.14 (PyInstaller cannot
cross-compile, so this one step has to happen there):

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

It leaves a zip in `%USERPROFILE%\StereoCraft-build`. Unzip it anywhere -- a USB
stick is fine -- and double-click `StereoCraft.exe` for the window, or run
`StereoCraft-cli.exe --help` for the command line. Nothing is installed and
nothing is downloaded on first run: the weights ship in `models\large` beside
the exe, so it works on a machine that has never seen the internet.

| switch | |
| --- | --- |
| `-Cuda` | build against CUDA Torch: about three times the size, and the difference between 20 seconds a photo and a tenth of one |
| `-Models small,base,large` | which checkpoints to ship; `large` alone by default |
| `-Work <dir>` | where to build, `%USERPROFILE%\StereoCraft-build` by default |
| `-SkipZip` | leave the folder without packing it |
| `-Python <path>` | which `python.exe` to build with; found on its own otherwise |
| `-TorchIndex cu130` | a different CUDA build of Torch. The default `cu126` runs on any driver from 525 up, where `cu130` wants 580 or newer -- and note there is no `cu128` wheel for Python 3.14 |

With `-SkipZip` the app is left in `dist-<flavour>\StereoCraft` under the build
folder, ready to copy wherever it is going to live. Move it somewhere of its own
before building again, because the next build overwrites that folder.

A build wants room to work in -- roughly 4 GB for the CPU one and 12 GB for
CUDA, most of it the environment being frozen. All of it is disposable
afterwards except `StereoCraft-build\models`, which is worth keeping so that a
rebuild does not fetch the weights all over again.

The CPU folder comes to about 1.9 GB -- nearly three quarters of it the large
model's weights -- and zips to 1.5 GB, since neither weights nor DLLs compress.
`-Models small` trades a little depth quality for a 700 MB app. `-Cuda` builds
5.4 GB instead, almost all of it CUDA kernels, and is worth every byte on a
machine with the card to use them: 0.1 s a photo against 21.8 s.

Either way the first photo of a session pays for loading the model, and on CUDA
the first run on a new machine pays again while the driver builds its kernel
cache -- 14 s once, then 3 s at the start of each session, then a tenth of a
second per photo.

The spec has not been brought forward to Depth Anything 3 either. It still
excludes OpenCV and torchvision, which the depth model now needs, so a packaged
build will want both added and will grow accordingly.

Windows has no way to know an unsigned exe, so the first run brings up a
SmartScreen box -- More info, then Run anyway -- and only a signing certificate
makes that go away.

## Use it

```bash
stereocraft photo.jpg
```

```bash
stereocraft ~/Pictures/holiday --output ~/Pictures/3d
```

```bash
stereocraft-gui
```

The window keeps the depth model resident, so the first photo pays the load and
the rest convert in well under a second each.

## When a photo is too big

Every conversion is sized up before the photo is even decoded, against what the
machine actually has free at that moment. A photo that will not fit in video
memory moves to the CPU, which usually has far more room. Only when it will not
fit there either is there a decision to make, and the app puts it to you rather
than guessing:

```
holiday.jpg is 11648x8736 (101.8 MP) and needs about 13.0 GB, but only 6.2 GB of memory is free.
Resizing to 7409x5556 (41.2 MP) would fit -- 64% of the width, 40% of the pixels.
The side-by-side image would come out about 14522x5556 instead of 22832x8736.
Resize and convert it, or skip it? [r/s]
```

The size offered is the largest one predicted to fit, so the detail given up is
the least that gets the photo converted. The window asks the same question in a
dialog. Nothing is ever downscaled without an answer: `--oversize skip` and a
non-interactive run both leave the photo alone and say so, and `--oversize
resize` takes the offer every time, for scripts that would rather have a
slightly smaller 3D photo than none.

The estimate is not calibrated to any particular machine. What a conversion
costs is worked out from the model you loaded, the precision it runs at and the
resolution it will run at, so a `small` model on a 4 GB card is judged on what
it actually needs rather than on what `large` would have needed. Free memory is
read from the machine itself -- the driver on CUDA and Apple silicon, the kernel
on Linux, Windows and macOS -- and both devices are priced, since a generous card
in a busy machine can have more to spare than the system does. When the depth
model alone is what does not fit, no resize can help, and it says which lighter
model would:

```
No smaller size would fit either: the depth model needs that much whatever size the photo is.
The base depth model would fit, and costs little in quality (--model base).
```

## Which settings?

None of them, most of the time: press Convert.

It used to be worth explaining why there was no correct setting to hunt for. A
photo carries no scale, the argument went, so the depth map says only what is
nearer than what and no combination of settings could reconstruct the real
geometry. That is no longer true, and the whole of this section is different
because of it.

The depth model measures in **metres**. So the renderer does not approximate
the separation between your eyes -- it works it out. Two eyes a real distance
apart, both looking at a screen a real distance away, see a point at distance Z
separated by

```
d = f · B · (1/Z − 1/Zc)
```

and that is what gets rendered. Things at the focus distance sit in the screen,
nearer things come out of it, and everything beyond recedes towards a **finite**
separation rather than being stretched out by however the depth map happened to
be scaled. That finite limit is the difference between geometry and a good guess,
and it is what makes a distant background sit properly behind the frame instead
of pulling apart.

### What auto does, and why it is not just the human number

The two settings are the eye separation and the focus distance, and by default
both are chosen per photo. It is worth knowing why, because "just use 65mm, the
human number" is the obvious answer and it is measurably wrong:

| Scene | Separation a real 65mm pair would need |
| --- | --- |
| a close-up 0.2–1.0 m away | 15.8% of frame width |
| a car 1.9–24.8 m away | 2.4% |
| a telephoto shot 19.4–22.5 m away | 0.3% |

Around 2% is comfortable. So literal eyes render a close-up no one can fuse and
a telephoto shot that is nearly flat. Both are *correct* -- that really is what
someone standing at the camera would see -- but you are not standing at the
camera, you are looking at a screen.

So the separation is chosen to suit the scene, which is what a stereographer
does rather than a fudge: a wider baseline than human for a distant landscape, a
narrower one for a close-up. On the nine photos above it picks 9mm for a macro,
56mm for a car at conversational distance, and 618mm for the telephoto shot, and
lands every one of them between 1.3% and 2.0%.

What matters is that only the amplitude moves. The *shape* stays exactly what
the metric geometry says -- parallax falling off as 1/Z, distant things
converging -- and that shape is the whole gain. Scaling it is a choice about how
big you want to feel, not an approximation.

- **Eye separation** (`--eyes`) — millimetres. `auto`, or a number: 65 for real
  human eyes, more for a landscape you want depth out of, less for a close-up.
- **Focus distance** (`--focus`) — metres. `auto`, or the distance you want
  sitting in the plane of the screen.
- **Target** (`--target`) — what `auto` aims for, as a percentage of frame width.
  The one to reach for if the whole thing is too strong or too flat: it keeps the
  geometry and changes only how much of it there is.

### Which model

`--model da3` is Depth Anything 3, and it is the only one that measures in
metres. `da2-large`, `da2-base` and `da2-small` are Depth-Anything V2, kept as a
fallback: they rank depth without measuring it, so their output is stretched onto
an assumed 1–50 m range and the geometry that comes out is approximate. Worth
reaching for if DA3 makes a mess of a particular photo, which does happen -- DA3's
own 1.4B variant was tested for this app and rejected because it falls apart on
portraits.

They are close on depth quality. On two test photos the edge-alignment scores
were 0.284 against 0.284 and 0.129 against 0.136 -- a wash. The reason to prefer
DA3 is the metres, not a sharper map.

### Where the metres come from

The conversion needs the lens. It is taken from the photo's EXIF where that
survives, and otherwise assumed to be a 28mm-equivalent, which is what most
phones point at the world.

Getting it wrong matters less than it sounds. A wrong focal length scales the
whole scene by the same factor, and `auto` then picks a baseline that cancels it
-- so the picture is unchanged and only the metre readings are off. It is worth
knowing before trusting a `--save-depth` map as a measurement.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `-e`, `--eyes` | `auto` | Distance between the two eyes, in millimetres, or `auto` to size it to the scene. 65 is the human average; a landscape wants far more and a close-up far less. |
| `-f`, `--focus` | `auto` | How far away the screen plane sits, in metres, or `auto`. Whatever is at this distance sits in the screen; nearer comes out, further recedes. |
| `-t`, `--target` | `2.0` | What `auto` aims for: near-to-far separation as a percentage of frame width. The knob to reach for when the effect is too strong or too flat. |
| `--limit` | `3.0` | Ceiling on separation, as a percentage of frame width, so something very close cannot demand more parallax than an eye can fuse. |
| `-m`, `--model` | `da3` | `da3` measures depth in metres. `da2-large`, `da2-base`, `da2-small` only rank it and are fitted onto an assumed range — a fallback, see [which model](#which-model). |
| `--depth-size` | `auto` | Longest side fed to the depth network (shortest, for the `da2` models). `auto` follows the photo up to 2048 px; bigger gives cleaner subject silhouettes, which is what the warp cares about. |
| `--cross` | off | Write right\|left for cross-eyed viewing instead of left\|right. Not a quality setting: it only matters when free-viewing on a monitor. |
| `--max-size` | `0` | Cap the output width. Native by default; useful if a viewer chokes on very wide images. |
| `--format`, `-q` | `auto`, `95` | Output container and JPEG quality. |
| `--save-depth` | off | Also write a 16-bit `_depth.png`, in centimetres — a map you can measure off rather than a grey ramp. Read [where the metres come from](#where-the-metres-come-from) before trusting it. |
| `--device` | `auto` | `cuda`, `mps` or `cpu`. |
| `--oversize` | `ask` | A photo too big for memory: `ask` what to do, `skip` it, or `resize` it to the largest size that fits. |

As a library:

```python
import stereocraft
stereocraft.convert("photo.jpg", eyes_mm=65, focus_m=3)
```

## How it works

1. **Depth** — Depth Anything 3 predicts depth in metres, converted to inverse
   depth (1/Z) because that is the quantity that behaves: it varies linearly
   across a slanted surface where depth itself does not, so the upsample below
   interpolates it correctly, and it is what the disparity formula wants anyway.
2. **Edge alignment** — the network runs at its own resolution and the result is
   lifted to full resolution with a guided filter that uses the photo as the
   guide. Depth edges land on picture edges instead of the soft ramps plain
   interpolation leaves, and that is what keeps silhouettes clean in the warp.
3. **Geometry** — two eyes are placed a real distance apart, aimed at a real
   focus distance, and each pixel's separation comes out as `f·B·(1/Z − 1/Zc)`.
   Both distances are chosen to suit the scene unless you pin them.
4. **Rendering** — each pixel is splatted into the column its disparity puts it
   in, with a z-buffer so nearer surfaces occlude, then resampled backwards with
   bilinear weights to recover sub-pixel detail.
5. **Disocclusions** — a small baseline only uncovers a few pixels of hidden
   background per edge. Those gaps are filled from whichever side is further
   away, with the background stretched gently across so there is no flat smear
   and no ghost of the foreground edge.
6. **Framing** — both eyes shift content sideways, leaving a sliver at the frame
   edge no real pixel reaches. It is trimmed rather than invented, costing about
   1% of the width.

There is no mesh, no inpainting network and no OpenGL context: the whole render
is a handful of tensor ops, which is why it runs at native resolution rather than
the 768px ceiling a mesh pipeline imposes.

## Viewing

The output is full-width SBS: each eye keeps its own full width, so the file is
twice as wide as the source. Quest, Pico and Vision Pro read it directly through
any local media viewer; on a desktop, free-viewing works with the parallel
method, or use `--cross` and cross your eyes.

## License

MIT, see [LICENSE](LICENSE).

The depth weights are downloaded at runtime rather than shipped here, and carry
their own terms: Depth-Anything V2 Small is Apache-2.0, while Base and Large —
Large being the default — are CC BY-NC 4.0, so they are not licensed for
commercial use.

That is worth a thought before handing the Windows build to anyone, since it
carries the weights inside it: a default build is not one to sell, and
`-Models small` makes an Apache-2.0 one that is.

This repository began as a fork of
[3D-Photo-Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) by way of
[Spatial-Photo](https://github.com/fake-oskars/Spatial-Photo). None of that code
remains; their notices stay with the commits that carried it.
