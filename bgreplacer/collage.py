"""Photo collage: built-in layouts, cell geometry and rendering.

A layout is a small character grid: each letter is one cell, rows are split
by "/". "AAB/AAC" = a big cell on the left and two small ones on the right.
Cells are numbered in reading order of their first appearance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import cos, lcm, pi, sin
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .pipeline import finish

MIN_PHOTOS, MAX_PHOTOS = 2, 9

# fonts that ship with Windows; nothing is bundled. Missing ones fall back below.
FONTS = {"Segoe UI": "seguisb.ttf", "Arial Black": "ariblk.ttf", "Georgia": "georgia.ttf",
         "Palatino": "pala.ttf", "Segoe Print": "segoepr.ttf", "Segoe Script": "segoesc.ttf"}
FALLBACK_FONTS = ("segoeui.ttf", "arial.ttf")
SHAPES = {"rect": "Rectangle", "circle": "Circle", "heart": "Heart", "hexagon": "Hexagon",
          "star": "Star"}
TEXT_COLORS = {"auto": None, "black": (20, 20, 20), "white": (255, 255, 255),
               "gold": (201, 162, 39)}


def rows(*counts: int) -> str:
    """Layout with the given number of equal cells per row, e.g. rows(2, 3)."""
    width = lcm(*counts)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return "/".join("".join(ch * (width // n) for ch in [next(letters) for _ in range(n)])
                    for n in counts)


def cols(*counts: int) -> str:
    """Layout with the given number of equal cells per column."""
    grid = rows(*counts).split("/")
    return "/".join("".join(r[i] for r in grid) for i in range(len(grid[0])))


LAYOUTS = {   # photo count -> layouts, the first is the default
    2: [rows(2), rows(1, 1), "AAB"],
    3: [rows(3), "AB/AC", "AA/BC", rows(1, 1, 1)],
    4: [rows(2, 2), "AAAB/AAAC/AAAD", "AAA/BCD", rows(4)],
    5: [rows(2, 3), "AABC/AADE", cols(2, 3)],
    6: [rows(3, 3), rows(2, 2, 2), "AAB/AAC/DEF"],
    7: [rows(3, 4), "AABC/AADE/FFGG", rows(2, 3, 2)],
    8: [rows(4, 4), rows(2, 2, 2, 2), rows(3, 2, 3)],
    9: [rows(3, 3, 3), "AABC/AADE/FGHI", rows(2, 3, 4)],
}


def cells(layout: str) -> list[tuple[float, float, float, float]]:
    """(l, t, r, b) fractions of each cell, in cell order."""
    grid = layout.split("/")
    nr, nc = len(grid), len(grid[0])
    boxes: dict[str, list[int]] = {}
    for y, line in enumerate(grid):
        for x, ch in enumerate(line):
            b = boxes.setdefault(ch, [x, y, x, y])
            b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)
    return [(l / nc, t / nr, (r + 1) / nc, (b + 1) / nr) for l, t, r, b in boxes.values()]


@dataclass
class Collage:
    layout: str
    views: list = field(default_factory=list)   # per cell: (zoom >= 1, pan x 0..1, pan y 0..1)
    order: list = field(default_factory=list)   # per cell: index of the photo shown
    aspect: float = 1.0                         # width / height
    spacing: int = 20                           # 0..100
    margin: int = 20                            # 0..100
    corners: int = 0                            # 0..100
    bg: tuple = (255, 255, 255)
    look: str = "none"
    vignette: int = 0
    grain: int = 0
    title: str = ""
    title_pos: str = "bottom"                   # top / bottom band, or "over" the photos
    font: str = "Segoe UI"
    text_size: int = 40                         # 0..100, title and captions
    text_color: str = "auto"                    # title colour, see TEXT_COLORS
    captions: list = field(default_factory=list)   # per photo (not per cell): text
    shapes: list = field(default_factory=list)     # per photo: a key of SHAPES

    def __post_init__(self):
        n = len(cells(self.layout))
        self.order = self.order or list(range(n))
        self.views = self.views or [(1.0, 0.5, 0.5)] * n


def canvas_size(aspect: float, long_side: int) -> tuple[int, int]:
    if aspect >= 1:
        return long_side, max(1, round(long_side / aspect))
    return max(1, round(long_side * aspect)), long_side


def title_px(c: Collage, size: tuple[int, int]) -> int:
    return round((0.025 + 0.06 * c.text_size / 100) * max(size))


def title_band(c: Collage, size: tuple[int, int]) -> int:
    """Height of the strip a top or bottom title takes from the photos."""
    return round(title_px(c, size) * 2.0) if c.title.strip() and c.title_pos != "over" else 0


def cell_rects(c: Collage, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Pixel boxes (x0, y0, x1, y1). Sizes are fractions of the long side, so
    the preview and the full-size picture look the same."""
    w, h = size
    m = c.margin / 100 * 0.08 * max(w, h)
    g = c.spacing / 100 * 0.04 * max(w, h) / 2           # half gap, on inner edges only
    band = title_band(c, size)
    top = m + (band if c.title_pos == "top" else 0)
    ah = max(1.0, h - 2 * m - band)                      # height left for the photos
    out = []
    for l, t, r, b in cells(c.layout):
        x0 = m + l * (w - 2 * m) + (g if l > 0 else 0)
        y0 = top + t * ah + (g if t > 0 else 0)
        x1 = m + r * (w - 2 * m) - (g if r < 1 else 0)
        y1 = top + b * ah - (g if b < 1 else 0)
        out.append((round(x0), round(y0), max(round(x0) + 1, round(x1)),
                    max(round(y0) + 1, round(y1))))
    return out


def visible_box(photo_size, cell_size, view) -> tuple[float, float, float, float]:
    """Part of the photo that fills the cell: cover-fit, then zoom and pan."""
    (pw, ph), (cw, ch), (zoom, px, py) = photo_size, cell_size, view
    s = max(cw / pw, ch / ph) * zoom
    vw, vh = cw / s, ch / s
    l, t = px * (pw - vw), py * (ph - vh)
    return l, t, l + vw, t + vh


def pan(photo_size, cell_size, view, dx: float, dy: float) -> tuple:
    """New view after dragging the photo by (dx, dy) cell pixels."""
    (pw, ph), (cw, ch), (zoom, px, py) = photo_size, cell_size, view
    s = max(cw / pw, ch / ph) * zoom
    free_x, free_y = pw - cw / s, ph - ch / s           # photo pixels the view can travel
    if free_x > 1e-6:
        px -= dx / s / free_x
    if free_y > 1e-6:
        py -= dy / s / free_y
    return zoom, min(1.0, max(0.0, px)), min(1.0, max(0.0, py))


def shape_points(shape: str) -> list | None:
    """Outline of a frame shape in a unit square; None for rect and circle."""
    if shape == "hexagon":
        return [(0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a))
                for a in (pi / 180 * (-90 + 60 * k) for k in range(6))]
    if shape == "star":
        return [(0.5 + r * cos(a), 0.54 + r * sin(a))
                for r, a in ((0.5 if k % 2 == 0 else 0.2, pi / 180 * (-90 + 36 * k))
                             for k in range(10))]
    if shape == "heart":
        ts = [2 * pi * k / 160 for k in range(160)]
        xs = [16 * sin(t) ** 3 for t in ts]
        ys = [-(13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)) for t in ts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return [((x - x0) / (x1 - x0), (y - y0) / (y1 - y0)) for x, y in zip(xs, ys)]
    return None


def _shape(c: Collage, idx: int) -> str:
    return c.shapes[idx] if idx < len(c.shapes) and c.shapes[idx] in SHAPES else "rect"


def _text(c: Collage, idx: int) -> str:
    return c.captions[idx].strip() if idx < len(c.captions) else ""


def _caption_px(size: tuple[int, int], c: Collage) -> int:
    return max(6, round(min(size) * (0.05 + 0.06 * c.text_size / 100)))


def frame_box(c: Collage, rect: tuple, idx: int) -> tuple[int, int, int, int]:
    """Where photo idx is placed inside its cell: the whole cell for a
    rectangle, else a centred square, leaving a strip below for a caption."""
    x0, y0, x1, y1 = rect
    if _shape(c, idx) == "rect":
        return rect
    strip = round(_caption_px((x1 - x0, y1 - y0), c) * 2) if _text(c, idx) else 0
    s = max(1, min(x1 - x0, y1 - y0 - strip))
    left, top = (x0 + x1 - s) // 2, y0 + (y1 - y0 - strip - s) // 2   # shape + text centred
    return left, top, left + s, top + s


def _shape_mask(size: tuple[int, int], shape: str, radius: float) -> Image.Image | None:
    if shape == "rect" and radius < 0.5:
        return None
    k = 3                                                # supersample for smooth edges
    w, h = size
    m = Image.new("L", (w * k, h * k), 0)
    d = ImageDraw.Draw(m)
    if shape == "rect":
        d.rounded_rectangle((0, 0, w * k - 1, h * k - 1), radius * k, fill=255)
    elif shape == "circle":
        d.ellipse((0, 0, w * k - 1, h * k - 1), fill=255)
    else:
        d.polygon([(x * (w * k - 1), y * (h * k - 1)) for x, y in shape_points(shape)], fill=255)
    return m.resize(size, Image.Resampling.LANCZOS)


@lru_cache(maxsize=32)
def font(name: str, px: int) -> ImageFont.FreeTypeFont:
    folder = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for f in (FONTS.get(name), *FALLBACK_FONTS):
        if f and (folder / f).is_file():
            return ImageFont.truetype(str(folder / f), px)
    return ImageFont.load_default(px)


def _fitted(draw: ImageDraw.ImageDraw, text: str, name: str, px: int, max_w: float):
    """The font at px, made smaller if the text would be wider than max_w."""
    f = font(name, px)
    w = draw.textlength(text, font=f)
    return font(name, max(6, int(px * max_w / w))) if w > max_w else f


def _caption(tile: Image.Image, text: str, c: Collage):
    """White caption on a soft dark fade at the bottom of a cell."""
    w, h = tile.size
    px = _caption_px((w, h), c)
    band = min(h, round(px * 2.6))
    fade = Image.linear_gradient("L").resize((w, band)).point(lambda v: v * 150 // 255)
    tile.paste((0, 0, 0), (0, h - band), fade)
    draw = ImageDraw.Draw(tile)
    draw.text((w / 2, h - px * 0.6), text, fill=(255, 255, 255), anchor="md",
              font=_fitted(draw, text, c.font, px, w * 0.9))


def _on_bg(c: Collage) -> tuple:
    """Dark or white text, whichever reads on the background colour."""
    r, g, b = c.bg
    return (20, 20, 20) if 0.3 * r + 0.59 * g + 0.11 * b > 140 else (255, 255, 255)


def _title(out: Image.Image, c: Collage):
    w, h = out.size
    m = c.margin / 100 * 0.08 * max(w, h)
    band, px = title_band(c, out.size), title_px(c, out.size)
    y = {"top": m + band / 2, "bottom": h - m - band / 2}.get(c.title_pos, h / 2)
    draw = ImageDraw.Draw(out)
    f = _fitted(draw, c.title, c.font, px, w * 0.9)
    color = TEXT_COLORS.get(c.text_color)
    if color is None:                          # auto: readable on what is behind it
        color = _on_bg(c) if band else (255, 255, 255)
    if not band:                               # over the photos: soft shadow for contrast
        shadow = Image.new("L", out.size, 0)
        ImageDraw.Draw(shadow).text((w / 2, y), c.title, fill=150, anchor="mm", font=f)
        out.paste((0, 0, 0), (0, 0), shadow.filter(ImageFilter.GaussianBlur(px / 6)))
    draw.text((w / 2, y), c.title, fill=color, anchor="mm", font=f)


def render(photos: list[Image.Image], c: Collage, long_side: int) -> Image.Image:
    size = canvas_size(c.aspect, long_side)
    out = Image.new("RGB", size, tuple(c.bg))
    draw = ImageDraw.Draw(out)
    for rect, idx, view in zip(cell_rects(c, size), c.order, c.views):
        x0, y0, x1, y1 = frame_box(c, rect, idx)
        photo, cw, ch, shape, text = photos[idx], x1 - x0, y1 - y0, _shape(c, idx), _text(c, idx)
        tile = photo.resize((cw, ch), Image.Resampling.LANCZOS,
                            box=visible_box(photo.size, (cw, ch), view))
        if text and shape == "rect":
            _caption(tile, text, c)
        out.paste(tile, (x0, y0), _shape_mask((cw, ch), shape, c.corners / 100 * 0.5 * min(cw, ch)))
        if text and shape != "rect":           # in the strip under the shape
            cx0, cy0, cx1, cy1 = rect
            px = _caption_px((cx1 - cx0, cy1 - cy0), c)
            draw.text(((cx0 + cx1) / 2, y1 + px), text, fill=_on_bg(c), anchor="mm",
                      font=_fitted(draw, text, c.font, px, (cx1 - cx0) * 0.9))
    if c.title.strip():
        _title(out, replace(c, title=c.title.strip()))
    if c.look != "none" or c.vignette or c.grain:
        out = Image.fromarray(finish(np.asarray(out), c.look, c.vignette, c.grain))
    return out
