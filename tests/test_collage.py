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
    from PySide6.QtWidgets import QApplication
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
    w.close()
