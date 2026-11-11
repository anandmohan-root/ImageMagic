# ImageMagic

ImageMagic is an offline Windows photo editor (the project folder and code are still named `BgReplacer`). You pick a main JPEG and an output folder, then click the action button. The subject is cut out once, and an editor opens with a live preview.

- **New background is optional.** Pick one and the button reads **Replace background**. Leave it empty, or remove it with ✕ on its card, and the button reads **Edit photo**: the photo keeps its own background, and every tool still works on it, for example a portrait-mode blur. For this the app builds a "clean plate", a copy of the background with the subject filled in from its surroundings, so blurring leaves no halo of the person's colours.
- **Output folder:** once you choose a folder, it stays the default for every photo until you close the app. Until then, it follows each photo's own folder.
- **Cut-out: Standard or Fine hair.** Standard gives crisp edges and suits most photos. Fine hair runs a second model (BiRefNet matting) and takes its soft, see-through alpha only in a band around the outline, where both models agree, so loose hair blends better. It needs one more model pass per photo, and both models stay loaded (about 1 GB more RAM).

Editor tools:

- **Adjust:** rotate the photo left or right in 90° steps, tilt the subject ±45°, set the subject's size (10–400%, or scroll the mouse wheel), and drag the subject to move it. **Center** puts the subject in the middle, and **Reset all** undoes every change.
- **Colors:** brightness, contrast and saturation (−100 to +100), set separately for the **Subject** and the **Background**, whether new or the photo's own. Use them to match the person's lighting to the scene.
- **Blend:** controls how the subject merges with the background.
  - **Edge softness** feathers the outline.
  - **Edge shift** moves the outline: left shrinks it to remove a halo of the old background, right grows it to bring back lost detail.
  - **Light wrap** lets the background's light spill slightly onto the subject's edges.
  - **Shadow** adds a soft shadow of the subject on the background, with **direction** (left or right) and **softness**. It suits a subject near a wall or backdrop; it is not a ground shadow under the feet.
- **Blur:** DSLR-style lens blur of the background; the subject stays sharp.
  - **Background blur** sets the overall strength.
  - **Depth blur** uses a depth model (Depth Anything V2 Small) to measure distance in the photo: things at the subject's distance stay sharp, and blur grows the farther away (or nearer) they are, as with a real lens. It works only when the photo keeps its own background.
  - **Focus falloff** adds blur that grows with distance from the **focus line** (0 = top of the picture, 100 = bottom). It imitates depth of field on a flat background.
  - **Bokeh shape** makes out-of-focus lights round or hexagonal, and **Highlight glow** sets how strongly they bloom.
- **Looks:** finishing for the whole picture, so subject and background read as one photo.
  - **Match lighting** pulls the subject's brightness, contrast and colour cast part of the way towards the background's.
  - **Look** applies one colour grade: Warm, Cool, Golden, Film or B&W.
  - **Vignette** darkens the corners, and **Grain** adds fine film grain.
- **Crop:** choose **Result** to crop the finished picture freely or to Original, 1:1, 4:5, 3:2 or 16:9. Choose **Main photo** to trim the original photo first, for example to remove an object at the edge; the result frame follows this crop.
- **Hold to see original:** press and hold to compare with the original photo.
- **Save** renders the result at full resolution and writes it. **Discard** writes nothing. Clicking the action button again with the same photos reopens the editor instantly, without cutting the subject out again.
- **Start new:** appears after Save or Discard. It clears the photo, the background, the cut-out choice and all edits, so the app is as it was when it opened. Only the output folder is kept.

Nothing touches the disk until you press **Save**.

- **Nothing leaves the computer.** The model ships inside the app folder. The app has no network code and sends no telemetry, and a test fails if anything opens a socket during a run.
- **The main image is never changed.** Inputs are opened read-only. The app never writes to the main image's path, and it never overwrites any file.
- The result is saved as `<main-name>_bg-replaced.jpg`, or `<main-name>_edited.jpg` when the original background was kept, at quality 95 with the main image's EXIF data. If that name is taken it uses `_1`, `_2`, and so on.

## How it works

1. Both images are loaded with their EXIF orientation applied and converted to RGB.
2. A salient-object model finds the subject at its native 1024×1024 input size. The model is BiRefNet, with ISNet as the fallback.
3. The mask is scaled up to full resolution and its edges are refined with a guided filter.
4. Edge colours are cleaned automatically. Semi-transparent edge pixels, such as hair, still carry the old background's colour; blur-fusion foreground estimation (Forte & Pitié, 2021) recovers their true colour, so no old-background fringe shows against the new one.
5. A new background is cover-fitted to the main image: scaled until it covers the frame, then centre-cropped. Without one, the photo's own clean plate is used. It is made by OpenCV inpainting under the subject only, so every other pixel stays exactly as photographed. The optional lens blur runs here, in linear light with a disc or hexagon kernel and highlight weighting. Focus falloff blends four blur strengths row by row.
6. Your edits are applied: rotation, photo crop, edge shift and softness, subject tilt, size and position (one affine warp of the cut-out), colours, match lighting (statistics transfer in Lab colour space), shadow, light wrap, the result crop, then the look, vignette and grain. The preview uses a downscaled copy. Save runs the same code on the full-size images.
7. The two images are alpha-blended and the result is saved.

Code: [bgreplacer/pipeline.py](bgreplacer/pipeline.py) (processing) and [bgreplacer/app.py](bgreplacer/app.py) (UI).

## Setup (once)

These steps also set the app up on another PC after cloning the repository. The repository holds only the source code. The Python environment, the models (about 2.1 GB) and the portable build are too large for GitHub and are recreated by the steps below.

Requires Python 3.11+ (tested with 3.13).

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Download the models. This is the only step that uses the network, about 2.1 GB:

```bash
.venv\Scripts\python scripts\fetch_models.py
```

This puts four models in `models\` and writes their checksums to `models\SHA256SUMS.txt`: `BiRefNet-general-epoch_244.onnx` (MIT), `isnet-general-use.onnx` (Apache-2.0), `BiRefNet-matting-epoch_100.onnx` (MIT repository; optional, for Fine hair) and `depth-anything-v2-small.onnx` (Apache-2.0; optional, for Depth blur). The app hides Fine hair and disables Depth blur when their model is missing. See [models/MODEL_LICENSES.txt](models/MODEL_LICENSES.txt).

## Run

**Recommended: the portable app.** Double-click **ImageMagic** on the desktop, or `dist\ImageMagic\ImageMagic.cmd`. The folder holds everything the app needs: a Python runtime, only the libraries the app uses, the code and the models. It works offline, needs no install, and you can copy the whole folder (about 2.5 GB) to another Windows PC. Its only programs are `python.exe` and `pythonw.exe`, signed by the Python Software Foundation, so Smart App Control lets it run. Tested on 2026-11-09: no block events. Rebuild it after code changes:

```bash
.venv\Scripts\python scripts\make_portable.py
```

The script copies this PC's Python install and the app's runtime libraries from `.venv`: PySide6-Essentials, onnxruntime, OpenCV, NumPy and Pillow, plus their dependencies. It downloads nothing. A `python313._pth` file isolates the bundled Python from `PYTHONPATH` and from other Python installs.

**From the project folder:** after Setup, run:

```bash
.venv\Scripts\pythonw main.py
```

**Smart App Control:** this laptop has Windows Smart App Control turned on. It decides about every unsigned or self-signed program with a cloud check, and the result changes over time:

| Date | Unsigned `BgReplacer.exe` | Self-signed `BgReplacer.exe` |
|---|---|---|
| 2026-11-08 | blocked | allowed |
| 2026-11-09 | blocked | **blocked** |

On 2026-11-09 Smart App Control also blocked pip's `pyinstaller.exe`. Microsoft documents that only certificates from its Trusted Root Program are accepted, so the self-signed `.exe` can't be relied on here.

The shortcut always works, because it starts the app with `pythonw.exe`, which is signed by the Python Software Foundation. The packaged `.exe` runs on PCs without Smart App Control, or when it's signed with a certificate from a trusted CA.

The first click loads the model, which takes a few seconds. After that the model stays loaded. On CPU, BiRefNet takes roughly 15–25 s per 12–24 MP photo. The app needs up to about 3.5 GB of RAM for a 12 MP photo and about 6 GB for a 24 MP photo, and drops back to about 1.6 GB between runs. If BiRefNet cannot load, the app falls back to ISNet.

**Optional GPU (DirectML):** run `pip uninstall onnxruntime`, then `pip install onnxruntime-directml`. The app uses DirectML automatically when it is available.

## Test

```bash
.venv\Scripts\python -m pytest
```

The tests cover cover-fit, alpha compositing, mask upscaling, EXIF orientation, output naming and no-overwrite, a full run that leaves the main image byte-identical, and offline behaviour:

- the socket guard trips if anything opens a socket;
- a full run with a tiny generated model opens no socket;
- a full run with the bundled real model opens no socket (skipped if the model has not been downloaded);
- the app package contains no network imports.

## Build (PyInstaller, folder build, signed)

```bash
powershell -ExecutionPolicy Bypass -File scripts\build_and_sign.ps1
```

This runs PyInstaller and then signs `BgReplacer.exe`:

- The first run creates a code-signing certificate, `CN=BgReplacer (local code signing)`, in your personal certificate store. It is valid for 5 years, and its private key cannot be exported. Later builds reuse it.
- The certificate is not added to any trusted store, so the .exe's Properties → Digital Signatures tab shows it as not trusted. That's expected. On this laptop, Smart App Control still blocks the self-signed `.exe` (see Run).
- Signing is offline: no timestamp server is contacted.

The unsigned equivalent is `.venv\Scripts\python -m PyInstaller --noconfirm BgReplacer.spec`. Use `python -m` because Smart App Control can block pip's `pyinstaller.exe` launcher.

The build goes to `dist\BgReplacer\`. Start `dist\BgReplacer\BgReplacer.exe`. The models are bundled inside `dist\BgReplacer\_internal\models\`, so you can copy the whole `dist\BgReplacer` folder to another Windows PC and run it with no Python and no internet.
