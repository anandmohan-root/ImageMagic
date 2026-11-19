"""Collage: layouts, geometry, rendering and the window flow."""
import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from bgreplacer import collage as C  # noqa: E402

RED, BLUE, GREEN = (220, 30, 30), (30, 30, 220), (30, 200, 30)


@pytest.mark.parametrize("n", range(C.MIN_PHOTOS, C.MAX_PHOTOS + 1))
def test_every_layout_tiles_the_frame(n):
    for layout in C.LAYOUTS[n]:
        boxes = C.cells(layout)
        assert len(boxes) == n
        grid = layout.split("/")
        assert len({len(r) for r in grid}) == 1, layout             # rectangular
        for ch in set(layout) - {"/"}:                               # every cell is a rectangle
            ys, xs = zip(*[(y, x) for y, r in enumerate(grid) for x, c in enumerate(r) if c == ch])
            area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
            assert area == len(xs), (layout, ch)
        assert abs(sum((r - l) * (b - t) for l, t, r, b in boxes) - 1.0) < 1e-9


def test_spacing_is_between_cells_and_margin_outside():
    c = C.Collage("AB", spacing=50, margin=50)
    (a, b) = C.cell_rects(c, (1000, 500))
    m, gap = round(0.5 * 0.08 * 1000), round(0.5 * 0.04 * 1000)
    assert a[0] == m and a[1] == m and b[2] == 1000 - m and b[3] == 500 - m
    assert abs((b[0] - a[2]) - gap) <= 1
    c0 = C.Collage("AB", spacing=0, margin=0)
    assert C.cell_rects(c0, (1000, 500)) == [(0, 0, 500, 500), (500, 0, 1000, 500)]


def halves(left, right, size=(400, 200)):
    img = Image.new("RGB", size, left)
    img.paste(right, (size[0] // 2, 0, size[0], size[1]))
    return img


def test_render_fills_cells_pans_and_zooms():
    photos = [halves(RED, BLUE), Image.new("RGB", (300, 300), GREEN)]
    c = C.Collage("AB", spacing=40, margin=40, bg=(0, 0, 0))
    out = np.asarray(C.render(photos, c, 800))
    assert out.shape == (800, 800, 3)
    assert tuple(out[5, 5]) == (0, 0, 0)                            # margin shows the background
    assert tuple(out[400, 400]) == (0, 0, 0)                        # so does the gap
    assert tuple(out[400, 600]) == GREEN                            # second cell
    # a tall cell shows the middle of a wide photo by default; panning left shows red only
    left = C.cell_rects(c, (800, 800))[0]
    mid = ((left[0] + left[2]) // 2, (left[1] + left[3]) // 2)
    c2 = C.Collage("AB", spacing=40, margin=40, views=[(1.0, 0.0, 0.5), (2.0, 0.5, 0.5)])
    out2 = np.asarray(C.render(photos, c2, 800))
    assert tuple(out2[left[1] + 5, left[0] + 5]) == RED and tuple(out2[mid[1], left[2] - 5]) == RED
    c3 = C.Collage("AB", spacing=40, margin=40, views=[(1.0, 1.0, 0.5), (1.0, 0.5, 0.5)])
    assert tuple(np.asarray(C.render(photos, c3, 800))[mid[1], left[0] + 5]) == BLUE


def test_pan_moves_the_view_and_stays_inside_the_photo():
    v = C.pan((400, 200), (100, 200), (1.0, 0.5, 0.5), dx=50, dy=0)  # drag right: see more left
    assert v[1] < 0.5 and v[2] == 0.5                                # no vertical room to pan
    assert C.pan((400, 200), (100, 200), (1.0, 0.5, 0.5), dx=1e6, dy=0)[1] == 0.0
    assert C.pan((400, 200), (100, 200), (1.0, 0.5, 0.5), dx=-1e6, dy=0)[1] == 1.0
    l, t, r, b = C.visible_box((400, 200), (100, 200), (2.0, 0.0, 0.0))
    assert (l, t) == (0.0, 0.0) and abs((r - l) - 50) < 1e-9 and abs((b - t) - 100) < 1e-9


def test_rounded_corners_show_the_background():
    photos = [Image.new("RGB", (100, 100), RED)] * 2
    out = np.asarray(C.render(photos, C.Collage("AB", spacing=0, margin=0, corners=100,
                                                bg=(0, 0, 0)), 400))
    assert tuple(out[0, 0]) == (0, 0, 0) and tuple(out[100, 100]) == RED


def test_look_applies_to_the_whole_collage():
    photos = [Image.new("RGB", (100, 100), RED)] * 2
    out = np.asarray(C.render(photos, C.Collage("AB", margin=0, spacing=0, look="bw"), 200))
    r, g, b = out[100, 50]
    assert r == g == b


def test_window_collage_flow(tmp_path):
    from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit
    from bgreplacer.app import MainWindow
    from tests.test_ui_smoke import wait_idle

    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    paths = []
    for i, colour in enumerate((RED, BLUE, GREEN)):
        p = src / f"p{i}.jpg"
        halves(colour, (200, 200, 200), (640, 480)).save(p, quality=95)
        paths.append(str(p))
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    w.enter_collage()
    assert w.pages.currentIndex() == 2 and w.views.currentIndex() == 1
    assert not w.collage_save.isEnabled()
    w.add_photos(paths[:1])
    wait_idle(app, w)
    assert len(w.photos) == 1 and not w.collage_save.isEnabled()
    w.add_photos(paths + [paths[0]])                   # duplicates are ignored
    wait_idle(app, w)
    assert len(w.photos) == 3 and w.collage.layout == C.LAYOUTS[3][0]
    assert len(w.layout_group.buttons()) == len(C.LAYOUTS[3])
    assert w.out_card.path == str(src)                 # follows the photos' folder
    w.out_card.accept_path(str(out))

    w.layout_group.buttons()[1].click()
    w.collage_sliders["corners"][0].setValue(30)
    w.on_cell_swap(0, 2)
    w.on_cell_pan(1, 30, 0)
    w.on_cell_zoom(1, 3)
    app.processEvents()
    assert w.collage.layout == C.LAYOUTS[3][1] and w.collage.order == [2, 1, 0]
    assert w.collage.views[1][0] > 1.0 and w.collage.corners == 30
    w.title_edit.setText("Holiday")
    w.title_pos_group.buttons()[0].click()             # Top
    rows = w.caption_host.findChildren(QLineEdit)
    assert len(rows) == 3
    rows[2].setText("Garden")
    assert w.collage.title == "Holiday" and w.collage.title_pos == "top"
    assert w.collage.captions == ["", "", "Garden"]
    picks = w.caption_host.findChildren(QComboBox)
    assert len(picks) == 3
    picks[0].setCurrentIndex(picks[0].findData("heart"))
    assert w.collage.shapes == ["heart", "rect", "rect"]
    assert list(out.iterdir()) == []                   # nothing written before Save

    w.save_collage()
    wait_idle(app, w)
    saved = list(out.iterdir())
    assert [f.name for f in saved] == ["p2_collage.jpg"]
    with Image.open(saved[0]) as im:
        assert max(im.size) == 3000
    w.save_collage()                                   # a second save never overwrites
    wait_idle(app, w)
    assert sorted(f.name for f in out.iterdir()) == ["p2_collage.jpg", "p2_collage_1.jpg"]
    assert before == {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}

    assert not w.collage_dirty
    w.close_collage()                                  # saved, so no question
    assert w.pages.currentIndex() == 0 and w.views.currentIndex() == 0 and w.photos == []
    assert w.collage_sliders["corners"][0].value() == 0
    assert w.title_edit.text() == "" and w.collage.title == "" and w.collage.captions == []
    assert w.collage.shapes == []
    w.close()


def test_title_band_takes_room_from_the_photos_and_draws_text():
    photos = [Image.new("RGB", (100, 100), RED)] * 2
    plain = C.Collage("AB", spacing=0, margin=0, bg=(255, 255, 255))
    assert C.cell_rects(plain, (800, 800))[0] == (0, 0, 400, 800)
    for pos in ("top", "bottom"):
        c = C.Collage("AB", spacing=0, margin=0, bg=(255, 255, 255), title="Hello", title_pos=pos)
        band = C.title_band(c, (800, 800))
        x0, y0, x1, y1 = C.cell_rects(c, (800, 800))[0]
        assert band > 0 and y1 - y0 == 800 - band and (y0 == band) == (pos == "top")
        out = np.asarray(C.render(photos, c, 800))
        strip = out[:band] if pos == "top" else out[800 - band:]
        assert strip.min() < 60                              # dark text on the light band
    over = C.Collage("AB", spacing=0, margin=0, title="Hello", title_pos="over")
    assert C.title_band(over, (800, 800)) == 0
    assert C.title_band(C.Collage("AB", title="   "), (800, 800)) == 0   # blank: no band
    out = np.asarray(C.render(photos, over, 800))
    assert (out[350:450].min(axis=2) > 230).any()            # white title over red photos


def test_captions_follow_their_photo_and_long_text_shrinks():
    photos = [Image.new("RGB", (100, 100), RED), Image.new("RGB", (100, 100), BLUE)]
    c = C.Collage("AB", spacing=0, margin=0, captions=["Beach", ""])
    out = np.asarray(C.render(photos, c, 800))
    assert (out[700:, :400].min(axis=2) > 230).any()         # caption text under photo 1
    assert not (out[700:, 400:].min(axis=2) > 230).any()     # photo 2 has none
    swapped = np.asarray(C.render(photos, C.Collage("AB", spacing=0, margin=0, order=[1, 0],
                                                    captions=["Beach", ""]), 800))
    assert (swapped[700:, 400:].min(axis=2) > 230).any()     # moved with its photo
    long = C.Collage("AB", spacing=0, margin=0, captions=["W" * 40, ""], text_size=100)
    out = np.asarray(C.render(photos, long, 800))
    assert not (out[700:, 0:2].min(axis=2) > 230).any() and not (out[700:, 398:400].min(axis=2) > 230).any()


def test_missing_font_falls_back():
    assert C.font("No such font", 20) is not None
    assert C.font("Georgia", 20).size == 20


@pytest.mark.parametrize("shape", ["circle", "heart", "hexagon", "star"])
def test_shape_frames_mask_the_photo(shape):
    photos = [Image.new("RGB", (100, 100), RED), Image.new("RGB", (100, 100), BLUE)]
    c = C.Collage("AB", spacing=0, margin=0, bg=(255, 255, 255), shapes=[shape])
    assert C.frame_box(c, (0, 0, 400, 800), 0) == (0, 200, 400, 600)    # centred square
    out = np.asarray(C.render(photos, c, 800))
    assert tuple(out[205, 2]) == (255, 255, 255)                         # corner: background
    assert tuple(out[420, 200]) == RED                                   # inside the shape
    assert tuple(out[5, 600]) == BLUE                                    # photo 2 stays a rectangle


def test_shape_moves_with_its_photo_and_caption_goes_below():
    photos = [Image.new("RGB", (100, 100), RED), Image.new("RGB", (100, 100), BLUE)]
    c = C.Collage("AB", spacing=0, margin=0, bg=(255, 255, 255), shapes=["circle"],
                  order=[1, 0], captions=["Beach"])
    out = np.asarray(C.render(photos, c, 800))
    assert tuple(out[5, 5]) == BLUE and tuple(out[205, 402]) == (255, 255, 255)
    x0, y0, x1, y1 = C.frame_box(c, (400, 0, 800, 800), 0)
    assert y1 < 800 and y0 + y1 < 800                                   # centred above the text
    assert out[y1:800, 400:800].min() < 60                               # dark text on white
    assert C.frame_box(C.Collage("AB", shapes=["bogus"]), (0, 0, 40, 80), 0) == (0, 0, 40, 80)
