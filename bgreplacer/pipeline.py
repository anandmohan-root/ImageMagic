"""Background replacement pipeline. Pure local computation, no network."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ORIENTATION_TAG = 0x0112
Progress = Callable[[int, str], None]


# ---------------------------------------------------------------- image I/O

def load_image(path: str | Path) -> tuple[Image.Image, Image.Exif, bytes | None]:
    """Open read-only, apply EXIF orientation, convert to RGB.

    Returns (rgb_image, exif, icc_profile). The file is fully read and closed.
    """
    with open(path, "rb") as f:          # read-only handle
        with Image.open(f) as im:
            im.load()
            exif = im.getexif()
            icc = im.info.get("icc_profile")
            rgb = ImageOps.exif_transpose(im).convert("RGB")
    return rgb, exif, icc


def unique_output_path(main_path: str | Path, out_dir: str | Path,
                       suffix: str = "_bg-replaced") -> Path:
    """<out_dir>/<main-stem><suffix>.jpg, then _1, _2 ... if taken."""
    stem = Path(main_path).stem + suffix
    out_dir = Path(out_dir)
    candidate = out_dir / f"{stem}.jpg"
    n = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{n}.jpg"
        n += 1
    return candidate


def save_jpeg(img: Image.Image, main_path: str | Path, out_dir: str | Path,
              exif: Image.Exif | None = None, icc: bytes | None = None,
              suffix: str = "_bg-replaced") -> Path:
    """Save as JPEG q95 with EXIF. Never overwrites, never writes to main_path."""
    main_resolved = Path(main_path).resolve()
    kwargs = {"quality": 95}
    if exif is not None:
        exif[ORIENTATION_TAG] = 1        # pixels are already upright
        kwargs["exif"] = exif.tobytes()
    if icc:
        kwargs["icc_profile"] = icc
    while True:
        out = unique_output_path(main_path, out_dir, suffix)
        if out.resolve() == main_resolved:
            raise RuntimeError("Refusing to write to the main image's path")
        try:
            with open(out, "xb") as f:   # exclusive create: fails if file exists
                img.save(f, format="JPEG", **kwargs)
            return out
        except FileExistsError:          # appeared between check and create
            continue


# -------------------------------------------------------------- compositing

def cover_fit(bg: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale bg so it covers size (w, h), then center-crop to exactly size."""
    w, h = size
    scale = max(w / bg.width, h / bg.height)
    nw, nh = max(w, round(bg.width * scale)), max(h, round(bg.height * scale))
    resized = bg.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return resized.crop((left, top, left + w, top + h))


def composite(fg: Image.Image, bg: Image.Image, alpha: np.ndarray) -> Image.Image:
    """out = fg*alpha + bg*(1-alpha). alpha: float HxW in [0, 1]."""
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    f = np.ascontiguousarray(np.asarray(fg, dtype=np.uint8))
    b = np.ascontiguousarray(np.asarray(bg, dtype=np.uint8))
    out = cv2.blendLinear(f, b, a, 1.0 - a)          # per-pixel weights, native speed
    return Image.fromarray(out, "RGB")


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Grey-guide guided filter (He et al.) using box filters."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1))
    mean_i, mean_p = box(guide), box(src)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * src) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def refine_mask(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    """Upscale a low-res mask to full size and snap its edges to the image."""
    w, h = image.size
    full = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    radius = max(2, round(max(w, h) / 500))
    return np.clip(guided_filter(grey, full, radius, 1e-3), 0.0, 1.0)


# -------------------------------------------------------------- segmentation

def models_dir() -> Path:
    base = getattr(sys, "_MEIPASS", None)   # PyInstaller bundle
    root = Path(base) if base else Path(__file__).resolve().parent.parent
    return root / "models"


def find_models(directory: Path | None = None) -> list[Path]:
    """Cut-out models in order of preference: BiRefNet general, then ISNet."""
    d = directory or models_dir()
    general = [m for m in sorted(d.glob("BiRefNet*.onnx")) if "matting" not in m.name.lower()]
    return general + sorted(d.glob("isnet*.onnx"))


def find_fine_model(directory: Path | None = None) -> Path | None:
    """The optional BiRefNet matting model, used for "Fine hair" edges."""
    found = sorted((directory or models_dir()).glob("BiRefNet*matting*.onnx"))
    return found[0] if found else None


def fuse_masks(general: np.ndarray, matting: np.ndarray) -> np.ndarray:
    """"Fine hair": keep the general model's decision about WHAT the subject
    is, and take the matting model's soft alpha only in the band around its
    outline, and only where the two models roughly agree. (On some pictures
    the matting model drops parts of the subject; there the general mask wins.)"""
    size = max(general.shape)
    solid = (general > 0.5).astype(np.uint8)

    def disc(frac):
        r = max(1, round(frac * size))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    core = cv2.erode(solid, disc(0.012)).astype(np.float32)          # surely subject
    outside = 1.0 - cv2.dilate(solid, disc(0.04)).astype(np.float32)  # surely background
    band = cv2.GaussianBlur(1.0 - core - outside, (0, 0), 0.004 * size)

    def nearby(where):                       # matting's average over nearby sure pixels
        sigma = 0.04 * size
        return cv2.GaussianBlur(matting * where, (0, 0), sigma) / \
            (cv2.GaussianBlur(where, (0, 0), sigma) + 1e-4)

    # agree where matting also says "subject" in the core and "background" outside
    agree = np.clip((nearby(core) - 0.5) / 0.3, 0.0, 1.0) * \
        np.clip((0.5 - nearby(outside)) / 0.3, 0.0, 1.0)
    return np.clip(general + band * agree * (matting - general), 0.0, 1.0)


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # Give inference memory back after each run instead of caching it
    # (BiRefNet otherwise keeps ~8 GB reserved; peak ~12 GB -> ~7 GB).
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), opts, providers=providers)


class DepthEstimator:
    """Monocular depth (Depth Anything V2 Small) with a local ONNX model."""

    def __init__(self, model_path: Path):
        self.path = Path(model_path)
        self.session = _session(self.path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.fixed = (w, h) if isinstance(w, int) and isinstance(h, int) else None

    @staticmethod
    def find(directory: Path | None = None) -> Path | None:
        found = sorted((directory or models_dir()).glob("depth*.onnx"))
        return found[0] if found else None

    def predict(self, image: Image.Image) -> np.ndarray:
        """Depth map in [0, 1], 1 = nearest, at the model's working size."""
        if self.fixed:
            size = self.fixed
        else:                                   # shorter side 518, both multiples of 14
            k = 518 / min(image.size)
            size = tuple(max(14, round(v * k / 14) * 14) for v in image.size)
        x = np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
            np.array([0.229, 0.224, 0.225], np.float32)
        d = np.asarray(self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0])
        d = d.reshape(d.shape[-2], d.shape[-1]).astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / (hi - lo + 1e-8)


def background_depth(depth: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """(depth of the scene with the subject filled in from its surroundings,
    the subject's own depth). The focus sits at the subject's depth."""
    a = cv2.resize(alpha, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    sel = a > 0.5
    focus = float(np.median(depth[sel])) if sel.any() else 1.0
    r = max(2, round(0.02 * max(depth.shape)))
    hole = cv2.dilate((a > 0.05).astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    filled = cv2.inpaint((depth * 255.0 + 0.5).astype(np.uint8), hole, 5, cv2.INPAINT_TELEA)
    return cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), 1.0), focus


class Segmenter:
    """Salient-object segmentation with a local ONNX model."""

    def __init__(self, model_path: Path):
        self.path = Path(model_path)
        self.kind = "isnet" if self.path.name.lower().startswith("isnet") else "birefnet"
        self.session = _session(self.path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.size = (w if isinstance(w, int) else 1024, h if isinstance(h, int) else 1024)

    @classmethod
    def load_default(cls, directory: Path | None = None) -> "Segmenter":
        errors = []
        for p in find_models(directory):
            try:
                return cls(p)
            except Exception as e:  # e.g. not enough memory: try the fallback
                errors.append(f"{p.name}: {e}")
        raise FileNotFoundError(
            f"No usable model in {directory or models_dir()}. "
            "Run scripts/fetch_models.py once during setup. " + "; ".join(errors))

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a float mask in [0, 1] at the model's native input size."""
        x = np.asarray(image.resize(self.size, Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255.0
        if self.kind == "isnet":
            x = x - 0.5
        else:
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
                np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: x})
        m = np.asarray(out[0])[0, 0]
        if self.kind == "birefnet":
            m = 1.0 / (1.0 + np.exp(-m))          # logits -> probability
        else:
            lo, hi = float(m.min()), float(m.max())
            m = (m - lo) / (hi - lo + 1e-8)
        return m.astype(np.float32)


# ------------------------------------------------------------------ editing

@dataclass(frozen=True)
class Edit:
    """User adjustments. Rects are (left, top, right, bottom) fractions.

    rot90     quarter turns clockwise applied to the main photo (and frame)
    src_crop  trims the main photo; the result frame follows the trimmed photo
    tilt      degrees clockwise, turns the subject around its own centre
    scale     subject size, 1.0 = as photographed
    offset    subject shift as a fraction of the frame width / height
    out_crop  crops the final picture
    subj_tone / bg_tone  (brightness, contrast, saturation), each -100..100
    feather   edge softness 0..100
    shift     edge shift -100 (shrink) .. 100 (grow)
    wrap      light wrap 0..100: new background light spilling onto the edges
    blur      background lens blur 0..100
    depth     extra blur growing with real distance from the subject, 0..100
              (needs the depth model and the photo's own background)
    falloff   extra blur growing with distance from the focus line, 0..100
    focus     focus line height, 0 (top) .. 100 (bottom)
    bokeh     aperture shape: "round" or "hex";  glow: highlight bloom 0..100
    shadow / shadow_dir / shadow_soft   subject's shadow on the background:
              strength 0..100, direction -100 (left) .. 100 (right), softness 0..100
    match     pull the subject's lighting towards the background's, 0..100
    look      colour grade for the whole picture (see LOOKS)
    vignette / grain   0..100, applied to the whole picture
    """
    rot90: int = 0
    src_crop: tuple = (0.0, 0.0, 1.0, 1.0)
    tilt: float = 0.0
    scale: float = 1.0
    offset: tuple = (0.0, 0.0)
    out_crop: tuple = (0.0, 0.0, 1.0, 1.0)
    subj_tone: tuple = (0, 0, 0)
    bg_tone: tuple = (0, 0, 0)
    feather: int = 0
    shift: int = 0
    wrap: int = 0
    blur: int = 0
    depth: int = 0
    falloff: int = 0
    focus: int = 100
    bokeh: str = "round"
    glow: int = 50
    shadow: int = 0
    shadow_dir: int = 40
    shadow_soft: int = 50
    match: int = 0
    look: str = "none"
    vignette: int = 0
    grain: int = 0

    def rotated(self, k: int) -> "Edit":
        """Turn the photo k quarter turns (positive = clockwise), keeping
        crops and subject position attached to the same content."""
        e = self
        for _ in range(k % 4):                      # one clockwise turn each
            (l, t, r, b), (L, T, R, B), (ox, oy) = e.src_crop, e.out_crop, e.offset
            e = replace(e, rot90=(e.rot90 + 1) % 4, src_crop=(1 - b, l, 1 - t, r),
                        out_crop=(1 - B, L, 1 - T, R), offset=(-oy, ox))
        return e


# Effect sizes are fractions of the frame's longer side, so the downscaled
# preview and the full-size save look the same.
EDGE_MAX = 0.004          # softness / shift at 100 -> 0.4 % of the long side
WRAP_SIGMA = 0.012        # light-wrap reach
BLUR_MAX = 0.03           # lens-blur radius at 100 -> 3 % of the long side


def _box(size: tuple[int, int], rect: tuple) -> tuple[int, int, int, int]:
    w, h = size
    l, t = int(round(rect[0] * w)), int(round(rect[1] * h))
    r, b = int(round(rect[2] * w)), int(round(rect[3] * h))
    l, t = min(max(l, 0), w - 1), min(max(t, 0), h - 1)
    return l, t, max(l + 1, min(r, w)), max(t + 1, min(b, h))


def tone(img, adj: tuple) -> np.ndarray:
    """Brightness, contrast, saturation in -100..100 (0 = unchanged).

    Brightness scales by 0.4x..1.6x, contrast stretches around mid-grey,
    saturation blends towards / away from the pixel's luminance.
    """
    x = np.asarray(img)
    if tuple(adj) == (0, 0, 0):
        return x
    b, c, s = adj
    x = x.astype(np.float32) / 255.0
    if b:
        x *= 1.0 + 0.6 * b / 100
    if c:
        x = (x - 0.5) * (1.0 + c / 100) + 0.5
    if s:
        grey = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = grey[..., None] + (x - grey[..., None]) * (1.0 + s / 100)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def estimate_foreground(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Remove the old background's colour from semi-transparent edge pixels.

    Blur-fusion foreground estimation (Forte & Pitie, "Approximate Fast
    Foreground Colour Estimation", ICIP 2021): two passes, a wide one then a
    narrow one, solving I = a*F + (1-a)*B for F with locally smoothed F and B.
    """
    x = np.asarray(img, dtype=np.float32) / 255.0
    a = alpha.astype(np.float32)[..., None]
    size = max(img.size)

    def fuse(f, b, r):
        k = (max(3, r), max(3, r))
        ba = cv2.blur(a, k)[..., None]
        bf = cv2.blur(f * a, k) / (ba + 1e-5)
        bb = cv2.blur(b * (1 - a), k) / ((1 - ba) + 1e-5)
        f = np.clip(bf + a * (x - a * bf - (1 - a) * bb), 0.0, 1.0)
        return f, bb

    full = x, a
    q = min(1.0, 1500 / size)                     # wide pass at reduced scale
    if q < 1.0:
        x = cv2.resize(full[0], None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
        a = cv2.resize(full[1][..., 0], None, fx=q, fy=q, interpolation=cv2.INTER_AREA)[..., None]
    f, b = fuse(x, x, round(0.06 * size * q))
    x, a = full
    if q < 1.0:
        hw = (x.shape[1], x.shape[0])
        f = cv2.resize(f, hw, interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, hw, interpolation=cv2.INTER_LINEAR)
    f, _ = fuse(f, b, round(0.004 * size))
    return Image.fromarray((f * 255.0 + 0.5).astype(np.uint8))


def aperture(r: float, shape: str = "round") -> np.ndarray:
    """Anti-aliased aperture kernel of radius r: a disc or a hexagon."""
    n = int(np.ceil(r))
    if shape == "hex":
        q = 4                                           # draw 4x larger, shrink: smooth edges
        side = (2 * n + 1) * q
        c = side / 2 - 0.5
        ang = np.deg2rad(np.arange(6) * 60 + 30)
        pts = np.stack([c + (r + 0.5) * q * np.cos(ang), c + (r + 0.5) * q * np.sin(ang)], 1)
        k = np.zeros((side, side), np.float32)
        cv2.fillConvexPoly(k, np.round(pts).astype(np.int32), 1.0)
        return cv2.resize(k, (2 * n + 1, 2 * n + 1), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return np.clip(r + 0.5 - np.sqrt(xx * xx + yy * yy), 0.0, 1.0).astype(np.float32)


# gamma tables: 8-bit sRGB-ish -> linear light, and back from 16-bit linear
_TO_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
_TO_GAMMA = (np.linspace(0.0, 1.0, 65536) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def lens_blur(img, radius: float, shape: str = "round", glow: int = 50) -> np.ndarray:
    """DSLR-style background blur: the aperture shape (disc or hexagon) shows
    in bright highlights, which bloom (bokeh); glow sets how strongly. Done in
    linear light; large radii are computed at a reduced scale, which looks
    identical because the detail is blurred away."""
    x = np.asarray(img)
    if radius < 0.5:
        return x
    h, w = x.shape[:2]
    f = min(1.0, 10.0 / radius)
    small = _TO_LINEAR[cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                       if f < 1 else x]
    r = radius * f
    disc = aperture(r, shape)
    lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    weight = 1.0 + 0.24 * glow * np.clip((lum - 0.6) / 0.4, 0.0, 1.0) ** 2   # highlights dominate
    num = cv2.filter2D(small * weight[..., None], -1, disc, borderType=cv2.BORDER_REFLECT)
    den = cv2.filter2D(weight, -1, disc, borderType=cv2.BORDER_REFLECT)
    out = num / den[..., None]
    out = _TO_GAMMA[(np.clip(out, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if f < 1 else out


def varied_blur(img, base: float, extra: float, amount: np.ndarray, shape: str = "round",
                glow: int = 50) -> np.ndarray:
    """Lens blur with a per-pixel radius: base + extra * amount, amount in [0, 1].
    Four blur strengths are computed and blended."""
    if extra < 0.5:
        return lens_blur(img, base, shape, glow)
    x = np.ascontiguousarray(np.asarray(img))
    t = 3.0 * np.clip(amount, 0.0, 1.0).astype(np.float32)
    out = np.ascontiguousarray(lens_blur(x, base, shape, glow))
    for i in (1, 2, 3):
        m = np.clip(t - (i - 1), 0.0, 1.0)
        if m.max() > 0:
            out = cv2.blendLinear(out, lens_blur(x, base + extra * i / 3, shape, glow),
                                  1.0 - m, m)
    return out


def focus_rows(h: int, focus: int) -> np.ndarray:
    """Per-row distance (0..1) from the focus line; focus 0 = top, 100 = bottom."""
    y0 = focus / 100 * (h - 1)
    return np.abs(np.arange(h, dtype=np.float32) - y0) / max(y0, h - 1 - y0, 1.0)


def focus_blur(img, base: float, extra: float, focus: int, shape: str = "round",
               glow: int = 50) -> np.ndarray:
    """Lens blur whose radius grows with vertical distance from a focus line:
    base at the line, base + extra at the farthest edge. Imitates depth of
    field on a flat background."""
    if extra < 0.5:
        return lens_blur(img, base, shape, glow)
    h, w = np.asarray(img).shape[:2]
    return varied_blur(img, base, extra, np.repeat(focus_rows(h, focus)[:, None], w, 1),
                       shape, glow)


def match_light(fg: np.ndarray, bg: np.ndarray, a: np.ndarray, amount: int) -> np.ndarray:
    """Pull the subject's brightness, contrast and colour cast part of the way
    towards the background's (Reinhard-style statistics transfer in Lab,
    applied as one lookup table per Lab channel)."""
    if not amount:
        return fg
    step = max(1, max(a.shape) // 400)                       # statistics from a subsample
    sel = a[::step, ::step] > 0.5
    if sel.sum() < 16:
        return fg

    def lab(x):
        return cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2LAB)

    subj = lab(fg[::step, ::step])[sel].astype(np.float32)
    back = lab(bg[::step, ::step]).reshape(-1, 3).astype(np.float32)
    t = amount / 100
    mu_s, mu_b = subj.mean(0), back.mean(0)
    gain = 1.0 + 0.5 * t * (np.clip(back[:, 0].std() / (subj[:, 0].std() + 1e-3), 0.75, 1.3) - 1.0)
    shift = t * np.array([0.5, 0.5, 0.5], np.float32) * (mu_b - mu_s)
    v = np.arange(256, dtype=np.float32)
    lut = np.stack([(v - mu_s[0]) * gain + mu_s[0] + shift[0], v + shift[1], v + shift[2]], 1)
    lut = np.clip(lut + 0.5, 0, 255).astype(np.uint8)[:, None, :]
    return cv2.cvtColor(cv2.LUT(lab(fg), lut), cv2.COLOR_LAB2RGB)


def cast_shadow(bg: np.ndarray, a: np.ndarray, strength: int, direction: int,
                soft: int) -> np.ndarray:
    """Darken the background with a soft, offset copy of the subject's shape."""
    if not strength:
        return bg
    size = max(a.shape)
    m = np.float32([[1, 0, direction / 100 * 0.05 * size], [0, 1, 0.012 * size]])
    sh = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
    sh = _soft_blur(sh, (0.004 + soft / 100 * 0.03) * size)
    keep = 1.0 - np.clip(0.75 * strength / 100 * sh, 0.0, 1.0)
    return cv2.multiply(np.ascontiguousarray(bg), cv2.merge([keep, keep, keep]),
                        dtype=cv2.CV_8U)


# look: (contrast, lift, (r, g, b) gains, saturation)
LOOKS = {
    "none": None,
    "warm": (1.00, 0.00, (1.06, 1.00, 0.90), 1.00),
    "cool": (1.00, 0.00, (0.92, 1.00, 1.08), 1.00),
    "golden": (1.06, 0.00, (1.10, 1.02, 0.82), 1.10),
    "film": (0.95, 0.05, (1.03, 1.00, 0.96), 0.85),
    "bw": (1.10, 0.00, (1.00, 1.00, 1.00), 0.00),
}


_GRAIN: dict = {}          # last noise field, keyed by picture size


def finish(img: np.ndarray, look: str = "none", vignette: int = 0, grain: int = 0) -> np.ndarray:
    """Whole-picture finishing, shared by subject and background so they read
    as one photo: colour grade, vignette, film grain."""
    out = np.ascontiguousarray(img)
    spec = LOOKS.get(look)
    if spec:
        contrast, lift, gains, sat = spec
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        x = ((x - 0.5) * contrast + 0.5) * (1.0 - lift) + lift
        lut = np.stack([np.clip(x * g, 0.0, 1.0) for g in gains], 1)
        out = cv2.LUT(out, (lut * 255.0 + 0.5).astype(np.uint8)[:, None, :])
        if sat != 1.0:
            grey = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            out = cv2.addWeighted(out, sat, cv2.merge([grey, grey, grey]), 1.0 - sat, 0)
    h, w = out.shape[:2]
    if vignette:
        yy, xx = np.mgrid[-1:1:128j, -1:1:128j].astype(np.float32)
        r = np.clip((np.sqrt(xx * xx + yy * yy) / np.sqrt(2.0) - 0.35) / 0.65, 0.0, 1.0)
        m = cv2.resize(1.0 - 0.6 * vignette / 100 * r * r * (3 - 2 * r), (w, h),
                       interpolation=cv2.INTER_LINEAR)
        out = cv2.multiply(out, cv2.merge([m, m, m]), dtype=cv2.CV_8U)
    if grain:
        f = min(1.0, 2000 / max(h, w))                  # grain size follows the picture size
        if _GRAIN.get("size") != (h, w):
            n = np.random.default_rng(7).standard_normal(
                (max(1, round(h * f)), max(1, round(w * f))), np.float32)
            if f < 1.0:
                n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
            _GRAIN.update(size=(h, w), noise=n)
        n = (_GRAIN["noise"] * (0.12 * grain)).astype(np.int16)
        out = cv2.add(out, cv2.merge([n, n, n]), dtype=cv2.CV_8U)
    return out


def shape_edges(a: np.ndarray, shift: int, feather: int) -> np.ndarray:
    """Shrink/grow the matte (shift) and soften it (feather)."""
    size = max(a.shape)
    if shift:
        r = max(1, round(abs(shift) / 100 * EDGE_MAX * size))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        a = cv2.erode(a, k) if shift < 0 else cv2.dilate(a, k)
    if feather:
        a = cv2.GaussianBlur(a, (0, 0), max(0.5, feather / 100 * EDGE_MAX * size))
    return a


def light_wrap(fg: np.ndarray, bg: np.ndarray, a: np.ndarray, amount: int) -> np.ndarray:
    """Let the new background's light spill a little onto the subject's edges."""
    if not amount:
        return fg
    spill = _soft_blur(1.0 - a, WRAP_SIGMA * max(a.shape)) * a     # inside, near edges
    m = np.clip(0.8 * amount / 100 * spill, 0.0, 1.0)
    soft_bg = _soft_blur(bg, WRAP_SIGMA * max(a.shape))
    return cv2.blendLinear(np.ascontiguousarray(fg), soft_bg, 1.0 - m, m)


def _soft_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Wide Gaussian blur computed at reduced scale (same look, much faster)."""
    f = min(1.0, 4.0 / sigma)
    if f >= 1.0:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    small = cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def clean_plate(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    """The photo's own background with the subject filled in from its
    surroundings, so blurring or moving the subject leaves no ghost or halo.
    The fill is computed at reduced size (it only shows under heavy blur or
    where the subject was moved away)."""
    size = max(img.size)
    f = min(1.0, 640 / size)
    small = cv2.resize(np.asarray(img), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    m = (cv2.resize(alpha, (small.shape[1], small.shape[0])) > 0.02).astype(np.uint8) * 255
    r = max(2, round(0.015 * max(small.shape)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    filled = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), r)            # hide inpainting streaks
    w, h = img.size
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    # fill only where the subject is; every other pixel stays exactly as photographed
    mf = (alpha > 0.02).astype(np.float32)
    return Image.fromarray(cv2.blendLinear(np.ascontiguousarray(np.asarray(img)), up,
                                           1.0 - mf, mf))


def _pivot(alpha: np.ndarray) -> tuple[float, float]:
    """Centre of the subject's bounding box, as fractions of width / height."""
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.5)
    if not len(xs):
        return 0.5, 0.5
    return (xs.min() + xs.max() + 1) / 2 / w, (ys.min() + ys.max() + 1) / 2 / h


@dataclass
class Prepared:
    """Result of the expensive step: upright photo, its alpha, the background.

    fg is the photo with edge colours cleaned (estimate_foreground); it is
    what gets composited. main stays untouched for "see original" and cropping.
    keep_bg: no new background was chosen; bg is the photo's own clean plate,
    the same size as main, and it turns and crops together with the photo.
    depth: with keep_bg and a depth model, the scene's depth (1 = nearest, any
    size, subject filled in); depth_focus is the subject's own depth.
    """
    main: Image.Image
    alpha: np.ndarray                 # float32 HxW in [0, 1]
    bg: Image.Image
    exif: Image.Exif | None = None
    icc: bytes | None = None
    fg: Image.Image | None = None
    keep_bg: bool = False
    depth: np.ndarray | None = None
    depth_focus: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def suffix(self) -> str:
        return "_edited" if self.keep_bg else "_bg-replaced"

    def preview(self, max_side: int) -> "Prepared":
        """Downscaled copy for fast interactive rendering."""
        main = self.main.copy()
        main.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        alpha = cv2.resize(self.alpha, main.size, interpolation=cv2.INTER_AREA)
        fg = self.fg.resize(main.size, Image.Resampling.BILINEAR) if self.fg else None
        if self.keep_bg:
            bg = self.bg.resize(main.size, Image.Resampling.BILINEAR)
        else:
            bg = self.bg.copy()
            bg.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return Prepared(main, alpha, bg, fg=fg, keep_bg=self.keep_bg, depth=self.depth,
                        depth_focus=self.depth_focus)

    def backdrop(self, edit: Edit, size: tuple[int, int]) -> np.ndarray:
        """Background for the frame (cover-fitted new one, or the photo's own),
        lens-blurred; cached for the last setting."""
        key = (size, edit.blur, edit.depth, edit.falloff, edit.focus, edit.bokeh, edit.glow) + \
            ((edit.rot90, tuple(edit.src_crop)) if self.keep_bg else ())
        if key not in self._cache:
            self._cache.clear()
            bg = self.photo(edit, "plate")[0] if self.keep_bg else cover_fit(self.bg, size)
            unit = BLUR_MAX * max(size) / 100
            by_rows = edit.falloff * unit
            by_depth = edit.depth * unit if self.depth is not None else 0.0
            extra = by_rows + by_depth
            amount = None
            if extra >= 0.5:                       # per-pixel share of the extra blur
                w, h = size
                amount = np.repeat(focus_rows(h, edit.focus)[:, None] * (by_rows / extra), w, 1)
                if by_depth:
                    amount += self.depth_distance(edit, size) * (by_depth / extra)
            self._cache[key] = varied_blur(bg, edit.blur * unit, extra, amount,
                                           edit.bokeh, edit.glow)
        return self._cache[key]

    def depth_distance(self, edit: Edit, size: tuple[int, int]) -> np.ndarray:
        """How far each background pixel is from the subject in depth, 0..1,
        turned and cropped like the photo."""
        d, f = self.depth, self.depth_focus
        span = max(f - float(d.min()), float(d.max()) - f, 1e-3)
        k = edit.rot90 % 4
        if k:
            d = np.ascontiguousarray(np.rot90(d, -k))
        l, t, r, b = _box((d.shape[1], d.shape[0]), edit.src_crop)
        d = cv2.resize(d[t:b, l:r], size, interpolation=cv2.INTER_LINEAR)
        return np.clip(np.abs(d - f) / span, 0.0, 1.0)

    def photo(self, edit: Edit, layer: str = "main") -> tuple[Image.Image, np.ndarray]:
        """A photo layer and the alpha after rotation and source crop.
        layer: "main" (original), "fg" (edge-cleaned) or "plate" (own background)."""
        img = {"fg": self.fg, "plate": self.bg if self.keep_bg else None}.get(layer) or self.main
        a = self.alpha
        k = edit.rot90 % 4
        if k:
            img = img.transpose([None, Image.Transpose.ROTATE_270,
                                 Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_90][k])
            a = np.ascontiguousarray(np.rot90(a, -k))
        box = _box(img.size, edit.src_crop)
        if box != (0, 0, *img.size):
            img = img.crop(box)
            a = a[box[1]:box[3], box[0]:box[2]]
        return img, a

    def output_size(self, edit: Edit) -> tuple[int, int]:
        """Pixel size render(edit) will have, without rendering."""
        w, h = self.main.size
        if edit.rot90 % 2:
            w, h = h, w
        l, t, r, b = _box((w, h), edit.src_crop)
        l, t, r, b = _box((r - l, b - t), edit.out_crop)
        return r - l, b - t

    def centering_offset(self, edit: Edit) -> tuple[float, float]:
        """Offset that puts the subject's centre in the middle of the final crop."""
        _, a = self.photo(edit)
        px, py = _pivot(a)
        l, t, r, b = edit.out_crop
        return (l + r) / 2 - px, (t + b) / 2 - py

    def render(self, edit: Edit, crop: bool = True) -> Image.Image:
        """Composite with all edits applied. crop=False skips the final crop."""
        img, a = self.photo(edit, "fg")
        w, h = img.size
        fg = np.asarray(img)
        a = shape_edges(a, edit.shift, edit.feather)
        if (edit.tilt, edit.scale, tuple(edit.offset)) != (0.0, 1.0, (0.0, 0.0)):
            px, py = _pivot(a)
            m = cv2.getRotationMatrix2D((px * w, py * h), -edit.tilt, edit.scale)
            m[0, 2] += edit.offset[0] * w
            m[1, 2] += edit.offset[1] * h
            fg = cv2.warpAffine(fg, m, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT)
            a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        bg = tone(self.backdrop(edit, (w, h)), edit.bg_tone)
        fg = match_light(tone(fg, edit.subj_tone), bg, a, edit.match)
        bg = cast_shadow(bg, a, edit.shadow, edit.shadow_dir, edit.shadow_soft)
        out = composite(light_wrap(fg, bg, a, edit.wrap), bg, a)
        if not crop:                               # crop tool view: whole frame, grade only
            return Image.fromarray(finish(np.asarray(out), edit.look))
        if tuple(edit.out_crop) != (0.0, 0.0, 1.0, 1.0):
            out = out.crop(_box(out.size, edit.out_crop))
        if (edit.look, edit.vignette, edit.grain) == ("none", 0, 0):
            return out
        return Image.fromarray(finish(np.asarray(out), edit.look, edit.vignette, edit.grain))


# ----------------------------------------------------------------- pipeline

def prepare(main_path: str | Path, bg_path: str | Path | None, segmenter: Segmenter,
            progress: Progress | None = None,
            depth_model: DepthEstimator | None = None,
            fine_model: Segmenter | None = None) -> Prepared:
    """Load the image(s) and cut out the subject. Writes nothing.
    bg_path None keeps the photo's own background; with a depth_model its
    depth is measured too, for depth-aware blur. fine_model (the matting
    model) adds soft hair edges to the cut-out."""
    report = progress or (lambda pct, msg: None)
    report(5, "Loading images")
    main, exif, icc = load_image(main_path)
    bg = load_image(bg_path)[0] if bg_path else None
    report(15, f"Segmenting ({segmenter.path.name})")
    mask = segmenter.predict(main)
    if fine_model is not None:
        report(45, "Tracing fine hair")
        mask = fuse_masks(mask, fine_model.predict(main))
    report(70, "Refining edges")
    alpha = refine_mask(mask, main)
    report(82, "Cleaning edge colours")
    fg = estimate_foreground(main, alpha)
    if bg is None:
        report(88, "Filling in the background")
        prep = Prepared(main, alpha, clean_plate(main, alpha), exif, icc, fg, keep_bg=True)
        if depth_model is not None:
            report(93, "Estimating depth")
            prep.depth, prep.depth_focus = background_depth(depth_model.predict(main), alpha)
        return prep
    report(95, "Preparing editor")
    return Prepared(main, alpha, bg, exif, icc, fg)


def replace_background(main_path: str | Path, bg_path: str | Path | None,
                       out_dir: str | Path,
                       segmenter: Segmenter, progress: Progress | None = None,
                       edit: Edit | None = None) -> Path:
    """Full run without the editor. Returns the new JPEG. main_path is only read."""
    report = progress or (lambda pct, msg: None)
    prep = prepare(main_path, bg_path, segmenter, report)
    report(90, "Saving")
    out = save_jpeg(prep.render(edit or Edit()), main_path, out_dir, prep.exif, prep.icc,
                    prep.suffix)
    report(100, f"Saved {out.name}")
    return out
