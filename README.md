# StereoCraft

Turn a single photo — or a video — into a **side-by-side 3D** one.

One job, done well: estimate a depth map, re-project the picture into a left and
a right eye view, and write the pair as one frame you can drop straight into a
headset or view free-eyed. A photo keeps every pixel it arrived with; a clip is
squeezed to half a frame per eye, because that is what players will decode.

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
clip.mp4   ->  clip_sbs.mp4         (left | right, half width per eye, sound kept)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps depth-anything-3
pip install -e . --no-deps
```

The `--no-deps` on the third line matters. Depth Anything 3's own dependency list
replaces Torch with a different CUDA build and pulls in a reconstruction and
visualisation stack — open3d, pycolmap, moviepy, flask, jupyter — that depth
inference never touches. `requirements.txt` carries what it actually reaches for
at runtime instead, each entry checked by blocking its import and converting a
photo.

The desktop window additionally needs Tkinter, which some Python builds omit:

```bash
sudo apt install python3-tk
```

JPEG, PNG, HEIC, WebP, TIFF and BMP all work, so photos come straight off a
phone without converting anything first.

Video needs ffmpeg, which does the decoding and encoding either side of the
conversion. A copy sitting next to the app is used if there is one, so the
portable build can carry its own:

```bash
sudo apt install ffmpeg
```

Depth weights download themselves on first run (~1.3 GB for the large model) and
are cached. If a `hf-cache/` folder exists next to the app it is used instead of
`~/.cache/huggingface`, so an existing download is picked up automatically.

## A Windows app

`packaging/windows/build.ps1` freezes the lot -- Python, Torch, the weights --
into one folder that runs on a Windows machine with nothing installed on it.
Build it on Windows, with any Python from 3.10 to 3.14 (PyInstaller cannot
cross-compile, so this one step has to happen there). Depth Anything 3's wheel
declares a ceiling of 3.13, which was just the newest version when it was
published; the build installs it past that, and it runs on 3.14 unchanged.

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

It leaves a zip in `%USERPROFILE%\StereoCraft-build`. Unzip it anywhere -- a USB
stick is fine -- and double-click `StereoCraft.exe` for the window, or run
`StereoCraft-cli.exe --help` for the command line. Nothing is installed and
nothing is downloaded on first run: the weights ship in `models\da3` beside the
exe and ffmpeg sits next to it, so both photos and video work on a machine that
has never seen the internet.

| switch | |
| --- | --- |
| `-Cuda` | build against CUDA Torch: about three times the size, and the difference between 20 seconds a photo and a tenth of one |
| `-Models da3,da2-large` | which checkpoints to ship; `da3` alone by default |
| `-SkipFfmpeg` | leave ffmpeg out, and with it video on a machine that has none |
| `-Work <dir>` | where to build, `%USERPROFILE%\StereoCraft-build` by default |
| `-SkipZip` | leave the folder without packing it |
| `-Python <path>` | which `python.exe` to build with; found on its own otherwise |
| `-TorchIndex cu130` | a different CUDA build of Torch. The default `cu126` runs on any driver from 525 up, where `cu130` wants 580 or newer. All three of `cu126`, `cu128` and `cu130` publish wheels for Python 3.14 |

The ffmpeg it fetches is the **LGPL** build, chosen so the finished folder can be
handed to anyone -- the usual "essentials" build is compiled `--enable-gpl` for
x264 and x265. The cost is that x264 is not in there, so video encodes on the
graphics card instead, or on a non-GPL software encoder. That is a little worse
per byte, and invisible unless you look. Copying a GPL `ffmpeg.exe` and
`ffprobe.exe` over the ones in the folder restores x264 immediately and needs no
flag -- worth doing for your own use, worth undoing before passing the folder on.
See [which encoder does the writing](#which-encoder-does-the-writing).

With `-SkipZip` the app is left in `dist-<flavour>\StereoCraft` under the build
folder, ready to copy wherever it is going to live. Move it somewhere of its own
before building again, because the next build overwrites that folder.

A build wants room to work in -- roughly 5 GB for the CPU one and 13 GB for
CUDA, most of it the environment being frozen. All of it is disposable
afterwards except `StereoCraft-build\models` and `StereoCraft-build\ffmpeg`,
worth keeping so that a rebuild does not fetch them all over again.

Depth Anything 3 made the folder bigger: OpenCV, torchvision, scipy and pandas
all have to come along now, on top of the 1.3 GB of weights. A CUDA build
measures 5.8 GB, against 5.4 GB before. The CPU one has not been rebuilt since
and its old figure of 1.9 GB will be low.

The first photo of a session pays for loading the model, and on CUDA the first
run on a new machine pays again while the driver builds its kernel cache.

One thing worth knowing about the frozen app: it runs with TorchScript turned
off. Depth Anything 3 puts `@torch.jit.script` on one small matrix helper, and
TorchScript compiles from source, which a frozen app does not carry -- so the
launcher disables the JIT before Torch is imported. The function then runs as
ordinary Python, which for a 4x4 inverse costs nothing measurable.

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
stereocraft clip.mp4
```

```bash
stereocraft-gui
```

Photos and clips go in the same run and the same window queue, and the depth
model is loaded once for the lot. The first photo pays the two second load and
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

## Video

```bash
stereocraft clip.mp4
```

Every frame gets exactly what a photo gets. Two things are added around it, and
both come from the picture moving rather than from anything about video files.

### The defaults are gentler

A clip aims for 1.3% of frame width where a photo aims for 2.0%, and pins the
depth network rather than letting it follow the frame.

Any error in the depth map becomes a horizontal position error in proportion to
the separation. In a still that is a silhouette a pixel out of place and nobody
sees it; in a clip it is an edge that shimmers and everybody does. The gaps the
warp opens up scale with it too, and a filled gap that reads as plausible while
it holds still crawls once the edge it belongs to moves. And a photo gets a
glance where a clip gets several minutes, so what is merely noticeable becomes
tiring.

`--target` overrides it in either direction, and is worth a try on your own
footage: 1.3 is a reasoned starting point, not a measured one.

The focus distance is left to `auto` exactly as it is for a photo. It puts most
of the scene behind the window with only a near subject in front of it, which is
the arrangement that stays comfortable — things poking out of the screen are what
break at the frame edge, and motion makes that worse rather than better.

### Depth is held still between frames

Depth Anything is a per-frame model, and a per-frame estimate wobbles. In a depth
map that reads as noise; turned into a stereo pair it is the *geometry* that
wobbles, which is a great deal harder to look at. `--temporal` carries some of
each frame's depth into the next, and a frame that differs wholesale from the one
before it is treated as a cut, where the memory starts again.

This section used to describe something more elaborate, and the honest version is
shorter. Depth-Anything V2 had to have its percentile range smoothed over time as
well, because re-measuring it every frame made the whole map slide about. Metric
depth needs no range at all — metres are metres whatever else is in the frame.

But that did not make things quieter, which is worth saying plainly because the
opposite was expected. Renormalising every frame had also been cancelling the
model's own scale wobble, and measured on a static shot with sensor noise, metric
depth is about a third *noisier* frame to frame. Smoothing the metric scale back
out was tried and made it worse still: the noise is spread through the map rather
than sitting in one global factor.

It does not matter, which is why only the plain average is left. In the units
that count — how far the disparity field actually moves between frames — both
models sit near a tenth of a pixel before any smoothing at all, against roughly a
third of a pixel for the smallest movement an eye can pick out:

| | no smoothing | `--temporal 0.5` |
| --- | --- | --- |
| Depth Anything 3 | 0.14 px | 0.07 px |
| Depth-Anything V2 | 0.11 px | 0.05 px |

Smoothing still costs a little edge sharpness, so `--temporal 0` declines it.

### Half a frame per eye

Unlike a photo, a clip does not come out at native width. Each eye is squeezed to
half the frame, so 1080p in is 1920×1080 out rather than 3792×1080.

That is what players and headsets expect, and more to the point what their
hardware decoders will take — 4K doubled is 7616 px wide, past the level h.264
defines and past what most headsets will decode at all. `--full` keeps every
native pixel for a player known to handle it, and `--codec hevc` is worth pairing
with it.

The soundtrack comes across untouched wherever the container will take it as it
stands, and is re-encoded to AAC only where it would otherwise be refused.
`--no-audio` leaves it behind.

### Which encoder does the writing

Not always the same one. `--codec` picks the format; what actually encodes it
depends on how the ffmpeg to hand was built, because x264 and x265 are GPL and
an LGPL ffmpeg carries neither. The order tried is:

1. **libx264 / libx265** — the best of them at a given file size
2. **NVENC**, then QSV, then AMF — the graphics card, if there is one
3. **libopenh264**, then MediaFoundation — software, and not GPL

Whichever it lands on, `--crf` still means quality, though the encoders spell it
differently underneath (`-crf`, `-cq`, `-global_quality`). Nothing is printed
about the choice, so if you want to know what wrote a file:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream_tags=encoder -of csv=p=0 clip_sbs.mp4
```

The portable Windows build ships an LGPL ffmpeg, so it uses the graphics card
rather than x264 — see [a Windows app](#a-windows-app). Dropping a GPL ffmpeg
into the folder puts x264 back, with no flag to set.

### How long it takes

| Clip | Per frame | Per minute of footage |
| --- | --- | --- |
| 1280 × 720 | 49 ms | 1.5 min |
| 1920 × 1080 | 58 ms | 1.7 min |
| 3840 × 2160 | 144 ms | 4.3 min |

On an RTX 4080 Super with the model already loaded — so roughly one and a half to
four times slower than watching it. The CPU is not really in the running: 1080p
costs 5.1 s a frame with the large model, which is two and a half hours for a
minute of footage. `--model base` brings that to 1.7 s and 52 minutes, which is
the difference between an overnight job and a hopeless one.

Anything that long is worth being able to watch and to stop. The command line
rewrites a line with the frame count and an estimate; the window shows the frame
it is working on and counts the queue down. Either can be stopped part-way, and
neither leaves a half-written file behind.

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
| `-t`, `--target` | `2.0` photo, `1.3` video | What `auto` aims for: near-to-far separation as a percentage of frame width. The knob to reach for when the effect is too strong or too flat. |
| `--limit` | `3.0` | Ceiling on separation, as a percentage of frame width, so something very close cannot demand more parallax than an eye can fuse. |
| `-m`, `--model` | `da3` | `da3` measures depth in metres. `da2-large`, `da2-base`, `da2-small` only rank it and are fitted onto an assumed range — a fallback, see [which model](#which-model). |
| `--depth-size` | `auto` photo, `1400` video | Longest side fed to the depth network (shortest, for the `da2` models). `auto` follows the photo up to 2048 px; bigger gives cleaner subject silhouettes, which is what the warp cares about. A clip is pinned, since the finer structure is what the temporal smoothing then averages away. |
| `--cross` | off | Write right\|left for cross-eyed viewing instead of left\|right. Not a quality setting: it only matters when free-viewing on a monitor. |
| `--max-size` | `0` | Cap the output width. Native by default; useful if a viewer chokes on very wide images. |
| `--format`, `-q` | `auto`, `95` | Output container and JPEG quality. |
| `--save-depth` | off | Also write a 16-bit `_depth.png`, in centimetres — a map you can measure off rather than a grey ramp. Read [where the metres come from](#where-the-metres-come-from) before trusting it. |
| `--device` | `auto` | `cuda`, `mps` or `cpu`. |
| `--oversize` | `ask` | A photo too big for memory: `ask` what to do, `skip` it, or `resize` it to the largest size that fits. |

Video only:

| Flag | Default | What it does |
| --- | --- | --- |
| `--temporal` | `0.5` | How much of the previous frame's depth to carry over, 0 to 0.95. Steadies a clip that shimmers, at a little edge sharpness; `0` turns it off. |
| `--full` | off | Keep every native pixel, doubling the frame width, instead of squeezing each eye to half width. Needs a player that will decode it. |
| `--codec` | `h264` | `hevc` is worth it above 4K, where h264 runs out of level. Which encoder produces it depends on the ffmpeg to hand — see [which encoder does the writing](#which-encoder-does-the-writing). |
| `--crf` | `18` | Encoder quality; lower is better and larger. Means the same thing whichever encoder runs, though they spell it differently underneath. |
| `--no-audio` | off | Leave the soundtrack behind rather than carrying it across. |

As a library:

```python
import stereocraft
stereocraft.convert("photo.jpg", eyes_mm=65, focus_m=3)
stereocraft.convert_video("clip.mp4", target_pct=1.0)
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
   1% of the width. A clip pins that trim to its settings rather than measuring
   it per frame, since frames that changed size part-way through are not
   something any encoder will take.
7. **Time**, for a clip only — the percentile range and then the depth map are
   carried forward from frame to frame, so the geometry stops wobbling. See
   [Video](#video).

There is no mesh, no inpainting network and no OpenGL context: the whole render
is a handful of tensor ops, which is why it runs at native resolution rather than
the 768px ceiling a mesh pipeline imposes.

## Viewing

A photo comes out full-width SBS: each eye keeps its own full width, so the file
is twice as wide as the source. A clip comes out half-width per eye, at the size
it went in, which is the arrangement players and their hardware decoders expect —
`--full` overrides that.

Quest, Pico and Vision Pro read both directly through any local media viewer; on
a desktop, free-viewing works with the parallel method, or use `--cross` and
cross your eyes.

Nothing in an mp4 announces that it is side-by-side, so players go by the file
name and most of them recognise the `_sbs` the output already carries. A player
that guesses wrong has a setting for it.

## License

MIT, see [LICENSE](LICENSE).

The depth weights are downloaded at runtime rather than shipped here, and carry
their own terms. The default is the permissive one, which it did not use to be:
**DA3METRIC-LARGE is Apache-2.0**, as is Depth-Anything V2 Small. Only the V2
Base and Large fallbacks are CC BY-NC 4.0, and those are not licensed for
commercial use — `-Models da2-large` puts one of them back in the folder.

Everything else in a default build is permissive too, which took some arranging:

| | |
| --- | --- |
| StereoCraft | MIT |
| DA3METRIC-LARGE weights, depth-anything-3, transformers, OpenCV, safetensors, huggingface-hub | Apache-2.0 |
| Torch, torchvision, NumPy, pandas, pillow-heif, OmegaConf | BSD |
| Pillow | MIT-CMU |
| imageio | BSD-2 |
| einops | MIT |
| tqdm | MPL-2.0 and MIT |
| ffmpeg (bundled) | LGPL v3 |

Two things were deliberately kept out. **`evo` is GPL-3.0** — DA3 imports it to
align camera poses, which a monocular app never does, so the module that reaches
for it is stubbed and `evo` is excluded from the build. And the ffmpeg that ships
is the **LGPL** build rather than the usual "essentials" one, which is compiled
`--enable-gpl` for x264 and x265. That costs the x264 encoder; the app falls to
the graphics card or to a non-GPL software encoder without being asked.

Dropping a GPL ffmpeg into the folder yourself restores x264 and is picked up
automatically — a reasonable thing to do for your own use, and a thing to think
about before redistributing.

`addict` states no licence in its metadata; worth confirming before you rely on
any of this. None of the above is legal advice.

This repository began as a fork of
[3D-Photo-Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) by way of
[Spatial-Photo](https://github.com/fake-oskars/Spatial-Photo). None of that code
remains; their notices stay with the commits that carried it.
