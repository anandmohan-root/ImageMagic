import numpy as np
from PIL import Image

from bgreplacer.pipeline import Edit, Prepared, composite, cover_fit


def make_prep():
    """80x40 photo, white 20x20 subject at x 10..30, y 10..30; blue background."""
    arr = np.zeros((40, 80, 3), np.uint8)
    arr[10:30, 10:30] = 255
    alpha = np.zeros((40, 80), np.float32)
    alpha[10:30, 10:30] = 1.0
    return Prepared(Image.fromarray(arr), alpha, Image.new("RGB", (100, 100), (0, 0, 255)))


def white(img):
    a = np.asarray(img)
    return (a[..., 0] > 200) & (a[..., 2] > 200)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_default_edit_equals_plain_composite():
    p = make_prep()
    expected = composite(p.main, cover_fit(p.bg, p.main.size), p.alpha)
    assert np.array_equal(np.asarray(p.render(Edit())), np.asarray(expected))


def test_rotate_90_swaps_frame_and_moves_subject():
    p = make_prep()
    out = p.render(Edit().rotated(1))
    assert out.size == (40, 80)
    # clockwise: top-left subject goes to the top-right
    x0, y0, x1, y1 = bbox(white(out))
    assert (x0, y0, x1, y1) == (10, 10, 29, 29)


def test_rotations_round_trip():
    e = Edit(src_crop=(0.1, 0.2, 0.7, 0.9), out_crop=(0, 0.5, 0.5, 1), offset=(0.1, -0.2))
    for back in (e.rotated(1).rotated(-1), e.rotated(4)):
        assert back.rot90 == e.rotated(4).rot90 == 0
        for f in ("src_crop", "out_crop", "offset"):
            assert np.allclose(getattr(back, f), getattr(e, f))


def test_source_crop_shrinks_frame():
    out = make_prep().render(Edit(src_crop=(0.0, 0.0, 0.5, 1.0)))
    assert out.size == (40, 40)


def test_output_crop():
    out = make_prep().render(Edit(out_crop=(0.25, 0.0, 0.75, 0.5)))
    assert out.size == (40, 20)


def test_move_shifts_subject():
    out = make_prep().render(Edit(offset=(0.25, 0.25)))        # +20 px, +10 px
    assert bbox(white(out)) == (30, 20, 49, 39)


def test_resize_scales_subject_around_its_centre():
    out = make_prep().render(Edit(scale=0.5))
    x0, y0, x1, y1 = bbox(white(out))
    assert abs((x1 - x0 + 1) - 10) <= 2 and abs((y1 - y0 + 1) - 10) <= 2
    assert abs((x0 + x1) / 2 - 19.5) <= 1.5                     # centre kept


def test_tilt_rotates_subject():
    out = make_prep().render(Edit(tilt=45))
    x0, _, x1, _ = bbox(white(out))
    assert x1 - x0 + 1 > 24                                      # diamond is wider


def test_preview_matches_full_render_shape():
    p = make_prep()
    e = Edit(rot90=1, out_crop=(0, 0, 1, 0.5))
    full, small = p.render(e), p.preview(40).render(e)
    assert abs(full.width / full.height - small.width / small.height) < 0.1


def test_tone_identity_and_ranges():
    from bgreplacer.pipeline import tone
    arr = np.array([[[200, 100, 50]]], np.uint8)
    assert np.array_equal(tone(arr, (0, 0, 0)), arr)
    assert tone(arr, (50, 0, 0)).mean() > arr.mean()          # brighter
    assert tone(arr, (-50, 0, 0)).mean() < arr.mean()         # darker
    grey = tone(arr, (0, 0, -100))[0, 0]
    assert grey.max() - grey.min() <= 1                       # fully desaturated
    flat = tone(arr, (0, -100, 0))[0, 0]
    assert all(abs(int(v) - 128) <= 1 for v in flat)          # no contrast -> mid-grey


def test_subject_tone_changes_subject_only():
    p = make_prep()
    p.main = Image.fromarray(np.full((40, 80, 3), 120, np.uint8))
    out = np.asarray(p.render(Edit(subj_tone=(50, 0, 0)))).astype(int)
    base = np.asarray(p.render(Edit())).astype(int)
    assert out[20, 20].sum() > base[20, 20].sum()             # subject brighter
    assert np.array_equal(out[2, 70], base[2, 70])            # background untouched


def test_background_tone_changes_background_only():
    p = make_prep()
    out = np.asarray(p.render(Edit(bg_tone=(0, 0, -100)))).astype(int)
    base = np.asarray(p.render(Edit())).astype(int)
    assert np.array_equal(out[20, 20], base[20, 20])          # subject untouched
    assert out[2, 70].max() - out[2, 70].min() <= 1           # blue bg -> grey


def test_edge_cleanup_removes_old_background_colour():
    from bgreplacer.pipeline import estimate_foreground
    # red subject on a green old background; the half-transparent edge column
    # is a 50/50 mix, so its true foreground colour is red
    h, w = 40, 80
    alpha = np.zeros((h, w), np.float32)
    alpha[:, :40] = 1.0
    alpha[:, 40] = 0.5
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :40] = (220, 0, 0)
    img[:, 40:] = (0, 200, 0)
    img[:, 40] = (110, 100, 0)
    f = np.asarray(estimate_foreground(Image.fromarray(img), alpha)).astype(int)
    assert f[20, 40, 1] < 40                      # green spill gone from the edge
    assert f[20, 40, 0] > 180                     # red restored
    assert np.abs(f[20, 10] - (220, 0, 0)).max() <= 2   # solid subject unchanged


def test_shift_and_feather():
    from bgreplacer.pipeline import shape_edges
    a = np.zeros((200, 200), np.float32)
    a[50:150, 50:150] = 1.0
    assert shape_edges(a, -100, 0).sum() < a.sum() < shape_edges(a, 100, 0).sum()
    soft = shape_edges(a, 0, 100)
    assert 0.05 < soft[100, 50] < 0.95            # edge became a gradient
    assert np.array_equal(shape_edges(a, 0, 0), a)


def test_light_wrap_tints_edges_only():
    p = make_prep()
    base = np.asarray(p.render(Edit())).astype(int)
    wrapped = np.asarray(p.render(Edit(wrap=100))).astype(int)
    # white subject, blue background: the edge pixel turns bluish (less red)
    assert wrapped[10, 10, 0] < base[10, 10, 0] and wrapped[10, 10, 2] == 255
    assert np.abs(wrapped[20, 20] - base[20, 20]).max() <= 1      # centre unchanged


def test_lens_blur_makes_bokeh_discs():
    from bgreplacer.pipeline import lens_blur
    img = np.zeros((101, 101, 3), np.uint8)
    img[50, 50] = 255                              # one bright point light
    out = lens_blur(img, 10)
    lit = out[..., 0] > 20
    ys, xs = np.nonzero(lit)
    assert 15 <= xs.max() - xs.min() + 1 <= 25     # a disc about 2r wide
    assert out[50, 50, 0] > 20 and out[50, 58, 0] > 20 and out[50, 64, 0] < 5
    assert np.array_equal(lens_blur(img, 0), img)


def test_background_blur_leaves_subject_sharp():
    p = make_prep()
    p.bg = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 100, 3), np.uint8))
    base = np.asarray(p.render(Edit())).astype(int)
    out = np.asarray(p.render(Edit(blur=100))).astype(int)
    assert np.array_equal(out[20, 20], base[20, 20])          # subject untouched
    assert out[:, 50:].std() < base[:, 50:].std()             # background smoother


def make_keep_prep():
    """Photo with its own textured background (no new background chosen)."""
    from bgreplacer.pipeline import clean_plate
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 120, (60, 80, 3)).astype(np.uint8)    # dark, textured scene
    arr[15:45, 25:55] = (240, 40, 40)                            # red subject
    alpha = np.zeros((60, 80), np.float32)
    alpha[15:45, 25:55] = 1.0
    img = Image.fromarray(arr)
    return Prepared(img, alpha, clean_plate(img, alpha), fg=img, keep_bg=True), arr


def test_clean_plate_removes_subject():
    p, arr = make_keep_prep()
    plate = np.asarray(p.bg).astype(int)
    assert plate[30, 40, 0] < 160                    # red subject filled in, not red
    assert np.array_equal(plate[2, 2], arr[2, 2])    # far background untouched


def test_keep_background_default_reproduces_photo():
    p, arr = make_keep_prep()
    out = np.asarray(p.render(Edit())).astype(int)
    assert out.shape == arr.shape
    assert np.abs(out - arr).mean() < 1.0
    assert p.suffix == "_edited"


def test_portrait_blur_blurs_own_background_not_subject():
    p, arr = make_keep_prep()
    out = np.asarray(p.render(Edit(blur=100))).astype(int)
    assert np.array_equal(out[30, 40], arr[30, 40])            # subject sharp
    assert out[:, :15].std() < arr[:, :15].std()                # background smoother
    assert out[30, 5, 0] < 160                                  # no red smeared outwards


def test_keep_background_follows_rotation_and_crop():
    p, _ = make_keep_prep()
    out = p.render(Edit(src_crop=(0.0, 0.0, 0.5, 1.0)).rotated(1))
    assert out.size == (60, 40)


def test_edited_suffix_naming(tmp_path):
    from bgreplacer.pipeline import save_jpeg
    out = save_jpeg(Image.new("RGB", (8, 8)), tmp_path / "a.jpg", tmp_path, suffix="_edited")
    assert out.name == "a_edited.jpg"
