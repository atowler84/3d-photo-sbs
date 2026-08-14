# sbs3d

Turn a single photo into a **full-resolution side-by-side 3D image**.

One job, done well: estimate a depth map, re-project the photo into a left and a
right eye view, and write the pair as one image you can drop straight into a
headset or view free-eyed. Nothing is downscaled along the way.

| Photo | Output | Time |
| --- | --- | --- |
| 1.9 MP | 3132 × 1197 | 0.3 s |
| 27 MP | 11784 × 4500 | 1.8 s |
| 60 MP | 18664 × 6336 | 4.1 s |

Measured on an RTX 4080 Super with the model already loaded; add about two
seconds for the first photo of a session. The GPU path falls back to the CPU
automatically if a photo will not fit in video memory, and offers to resize a
photo that will not fit there either -- see [when a photo is too
big](#when-a-photo-is-too-big).

There is a CPU path too, and it is a lot slower: the same three photos take
20 s, 24 s and 34 s on an eight-core Ryzen 7 7800X3D. Most of that is a fixed
cost rather than a per-pixel one, because the depth network always runs at a
518-1036 px short side no matter how big the photo is, so a snapshot costs
almost as much as a 60 MP raw. `--model base` brings those three down to 7 s,
12 s and 20 s for slightly softer depth edges, which is the trade worth making
on a machine without a GPU.

```
photo.jpg  ->  photo_sbs.jpg        (left | right, full width, no downscaling)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The desktop window additionally needs Tkinter, which some Python builds omit:

```bash
sudo apt install python3-tk
```

JPEG, PNG, HEIC, WebP, TIFF and BMP all work, so photos come straight off a
phone without converting anything first.

Depth weights download themselves on first run (~1.3 GB for the large model) and
are cached. If a `hf-cache/` folder exists next to the app it is used instead of
`~/.cache/huggingface`, so an existing download is picked up automatically.

## Use it

```bash
sbs3d photo.jpg
```

```bash
sbs3d ~/Pictures/holiday --output ~/Pictures/3d
```

```bash
sbs3d-gui
```

The window keeps the depth model resident, so the first photo pays the two second
load and the rest convert in well under a second each.

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

It is worth knowing why there is no "correct" setting to hunt for. A single photo
carries no scale. The depth map says what is nearer than what, never how far away
anything is in metres, so no combination of settings reconstructs the real
geometry of the scene. What you are choosing is how much depth to portray, and
the defaults are picked to look natural and stay comfortable over a long look.

Two things are genuinely a matter of taste, and they are the only two the window
puts in front of you:

- **Depth strength** (`--disparity`) - how far apart your two virtual eyes are.
  Past about 3% the scene starts reading as flat cards at different distances,
  and it gets tiring to look at.
- **Screen plane** (`--convergence`) - which depth sits at the window. At the
  default only your nearest subject comes forward and the rest of the scene sits
  behind the frame, which is the restful arrangement. Lower it to push everything
  further back.

Everything else the window decides for you: always the largest depth model, always
the best working resolution for the photo you gave it. The command line keeps
`--model` and `--depth-size` for scripting and slow machines, but there is no
quality reason to touch them.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `-d`, `--disparity` | `2.0` | Eye separation as a percentage of image width. 1.5–3 is comfortable; past 4 the depth reads as a cardboard cut-out. |
| `-c`, `--convergence` | `0.9` | Which depth sits on the screen plane, 0 (far) to 1 (near). The default parks almost everything behind the screen with only the nearest subject popping forward, which is the easy-on-the-eyes choice. |
| `-m`, `--model` | `large` | Depth-Anything V2 size: `small` (95 MB), `base` (372 MB), `large` (1.3 GB). Only worth lowering on a slow machine. |
| `--depth-size` | `auto` | Shorter side fed to the depth network. `auto` follows the photo between 518 and 1036 px — big photos get the finer structure, small ones do not pay for detail that is not there. Pin a number to override. |
| `--cross` | off | Write right\|left for cross-eyed viewing instead of left\|right. Not a quality setting: it only matters when free-viewing on a monitor. |
| `--max-size` | `0` | Cap the output width. Native by default; useful if a viewer chokes on very wide images. |
| `--format`, `-q` | `auto`, `95` | Output container and JPEG quality. |
| `--save-depth` | off | Also write a 16-bit `_depth.png`. |
| `--device` | `auto` | `cuda`, `mps` or `cpu`. |
| `--oversize` | `ask` | A photo too big for memory: `ask` what to do, `skip` it, or `resize` it to the largest size that fits. |

As a library:

```python
import sbs3d
sbs3d.convert("photo.jpg", disparity=2.5)
```

## How it works

1. **Depth** — Depth-Anything V2 predicts relative inverse depth, which is
   already proportional to stereo disparity, so it maps onto eye separation
   directly. Depth is normalised on the 2nd/98th percentiles so a blown-out sky
   or a speck of flare cannot squash the range.
2. **Edge alignment** — the network runs at its own resolution and the result is
   lifted to full resolution with a guided filter that uses the photo as the
   guide. Depth edges land on picture edges instead of the soft ramps plain
   interpolation leaves, and that is what keeps silhouettes clean in the warp.
3. **Rendering** — each pixel is splatted into the column its disparity puts it
   in, with a z-buffer so nearer surfaces occlude, then resampled backwards with
   bilinear weights to recover sub-pixel detail.
4. **Disocclusions** — a small baseline only uncovers a few pixels of hidden
   background per edge. Those gaps are filled from whichever side is further
   away, with the background stretched gently across so there is no flat smear
   and no ghost of the foreground edge.
5. **Framing** — both eyes shift content sideways, leaving a sliver at the frame
   edge no real pixel reaches. It is trimmed rather than invented, costing about
   1% of the width.

There is no mesh, no inpainting network and no OpenGL context: the whole render
is a handful of tensor ops, which is why it runs at native resolution rather than
the 768px ceiling a mesh pipeline imposes.

## Viewing

The output is full-width SBS (each eye keeps its own full width, so the file is
twice as wide as the source). Quest, Pico and Vision Pro read it directly through
any local media viewer; on a desktop, free-viewing works with the parallel
method, or use `--cross` and cross your eyes.

## License

MIT, see [LICENSE](LICENSE).

The depth weights are downloaded at runtime rather than shipped here, and carry
their own terms: Depth-Anything V2 Small is Apache-2.0, while Base and Large —
Large being the default — are CC BY-NC 4.0, so they are not licensed for
commercial use.

This repository began as a fork of
[3D-Photo-Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) by way of
[Spatial-Photo](https://github.com/fake-oskars/Spatial-Photo). None of that code
remains; their notices stay with the commits that carried it.
