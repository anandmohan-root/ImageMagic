"""Finishing effects: match lighting, shadow, looks, vignette, grain, focus falloff, bokeh."""
import numpy as np
from PIL import Image

from bgreplacer.pipeline import (Edit, Prepared, aperture, cast_shadow, finish, focus_blur,
                                 lens_blur, match_light)


def square_alpha(h=80, w=120):
    a = np.zeros((h, w), np.float32)
    a[20:60, 40:80] = 1.0
    return a


def test_match_light_moves_subject_towards_background():
    a = square_alpha()
    fg = np.full((80, 120, 3), 110, np.uint8)                    # neutral grey subject
    bg = np.zeros((80, 120, 3), np.uint8)
    bg[:] = (240, 200, 120)                                      # bright, warm scene
    out = match_light(fg, bg, a, 100).astype(int)
    px = out[40, 60]
    assert px.mean() > 120                                       # brighter
    assert px[0] > px[2] + 10                                    # warmer: red above blue
    assert px.mean() < 200                                       # only part of the way
    assert match_light(fg, bg, a, 0) is fg


def test_shadow_falls_to_the_chosen_side_only_on_background():
    a = square_alpha()
    bg = np.full((80, 120, 3), 200, np.uint8)
    out = cast_shadow(bg, a, 100, 100, 20).astype(int)
    assert out[40, 84, 0] < out[40, 36, 0] - 20                  # right of subject darker than left
    assert out[2, 2, 0] >= 198                                   # far corner untouched
    assert cast_shadow(bg, a, 0, 100, 20) is bg
    left = cast_shadow(bg, a, 100, -100, 20).astype(int)
    assert left[40, 36, 0] < left[40, 84, 0] - 20


def test_looks():
    rng = np.random.default_rng(0)
    img = rng.integers(30, 220, (40, 60, 3)).astype(np.uint8)
    assert np.array_equal(finish(img), img)
    bw = finish(img, "bw").astype(int)
    assert (bw.max(2) - bw.min(2)).max() <= 1
    warm = finish(img, "warm").astype(int)
    assert warm[..., 0].mean() > img[..., 0].mean() and warm[..., 2].mean() < img[..., 2].mean()
    film = finish(np.zeros((4, 4, 3), np.uint8), "film")
    assert film.min() > 5                                        # lifted blacks


def test_vignette_darkens_corners_not_centre():
    img = np.full((100, 150, 3), 200, np.uint8)
    out = finish(img, vignette=100).astype(int)
    assert out[50, 75, 0] >= 199
    assert out[0, 0, 0] < 120


def test_grain_is_subtle_and_repeatable():
    img = np.full((100, 150, 3), 128, np.uint8)
    a, b = finish(img, grain=50), finish(img, grain=50)
    assert np.array_equal(a, b)                                  # preview and save agree
    assert 2 < a.astype(float).std() < 12
    assert abs(a.mean() - 128) < 1
    assert (a.max(2) - a.min(2)).max() == 0                      # luminance grain, no colour specks


def test_hexagon_aperture_and_glow():
    disc, hexa = aperture(12, "round"), aperture(12, "hex")
    assert 0.75 < hexa.sum() / disc.sum() < 0.95                 # hexagon inside the circle
    img = np.full((101, 101, 3), 40, np.uint8)
    img[50, 50] = 255                                            # a point light
    dim, bright = lens_blur(img, 8, glow=0), lens_blur(img, 8, glow=100)
    assert int(bright[50, 54, 0]) > int(dim[50, 54, 0]) + 5      # glow strengthens the disc
    hexed = lens_blur(img, 8, "hex", 100)
    assert int(hexed[50, 58, 0]) < int(bright[50, 58, 0])        # flat side: cut where the disc reaches


def test_focus_falloff_keeps_the_focus_line_sharp():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (120, 90, 3)).astype(np.uint8)
    out = focus_blur(img, 0, 6, 100)                             # focus at the bottom
    assert np.array_equal(out[-1], img[-1])
    assert out[:10].astype(float).std() < 0.5 * img[:10].astype(float).std()
    mid = focus_blur(img, 0, 6, 50)
    assert mid[55:65].astype(float).std() > mid[:10].astype(float).std()
    assert np.array_equal(focus_blur(img, 3, 0, 50), lens_blur(img, 3))


def test_render_with_all_effects():
    arr = np.random.default_rng(2).integers(0, 255, (80, 120, 3)).astype(np.uint8)
    p = Prepared(Image.fromarray(arr), square_alpha(), Image.new("RGB", (200, 200), (90, 140, 200)),
                 fg=Image.fromarray(arr))
    e = Edit(shadow=80, match=60, look="film", vignette=50, grain=30, blur=40, falloff=50,
             focus=70, bokeh="hex", glow=80, out_crop=(0.0, 0.0, 0.5, 1.0))
    assert p.render(e).size == (60, 80)
    assert p.render(e, crop=False).size == (120, 80)
    base = np.asarray(p.render(Edit())).astype(int)
    lit = np.asarray(p.render(Edit(match=100))).astype(int)
    assert np.array_equal(lit[2, 2], base[2, 2])                 # match lighting leaves bg alone
    assert not np.array_equal(lit[40, 60], base[40, 60])
