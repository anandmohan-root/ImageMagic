import numpy as np
from PIL import Image

from bgreplacer.pipeline import composite, cover_fit, load_image, refine_mask


def test_cover_fit_exact_size_and_center_crop():
    bg = Image.new("RGB", (300, 100))
    bg.paste((255, 0, 0), (0, 0, 100, 100))       # red left third
    bg.paste((0, 255, 0), (100, 0, 200, 100))     # green middle
    out = cover_fit(bg, (100, 100))
    assert out.size == (100, 100)
    assert out.getpixel((50, 50)) == (0, 255, 0)  # centre kept


def test_cover_fit_upscales_small_background():
    out = cover_fit(Image.new("RGB", (10, 20)), (400, 300))
    assert out.size == (400, 300)


def test_composite_uses_alpha():
    fg = Image.new("RGB", (4, 1), (200, 200, 200))
    bg = Image.new("RGB", (4, 1), (0, 0, 100))
    alpha = np.array([[1.0, 0.0, 0.5, 0.25]], np.float32)
    px = np.asarray(composite(fg, bg, alpha))[0]
    assert tuple(px[0]) == (200, 200, 200)
    assert tuple(px[1]) == (0, 0, 100)
    assert tuple(px[2]) == (100, 100, 150)
    assert tuple(px[3]) == (50, 50, 125)


def test_refine_mask_upscales_to_image_size_in_range():
    img = Image.new("RGB", (123, 77), (128, 128, 128))
    mask = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    full = refine_mask(mask, img)
    assert full.shape == (77, 123)
    assert full.min() >= 0.0 and full.max() <= 1.0


def test_load_image_applies_exif_orientation(tmp_path):
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                               # rotate 90 CW on display
    Image.new("RGB", (40, 20)).save(p, exif=exif.tobytes())
    img, _, _ = load_image(p)
    assert img.size == (20, 40) and img.mode == "RGB"
