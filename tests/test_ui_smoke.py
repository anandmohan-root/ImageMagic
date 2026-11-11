"""Window flow: inputs -> editor -> Discard writes nothing; Save writes one file."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from bgreplacer import pipeline  # noqa: E402
from bgreplacer.app import MainWindow  # noqa: E402


def wait_idle(app, w, timeout=30):
    end = time.time() + timeout
    while w.busy() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_window_flow(jpegs, tiny_model):
    main, bg, out = jpegs
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    assert not w.run_btn.isEnabled()
    w.main_card.accept_path(str(main))
    assert w.out_card.path == str(main.parent)      # defaults to the photo's folder
    w.bg_card.accept_path(str(bg))
    w.out_card.accept_path(str(out))
    assert w.run_btn.isEnabled()

    w.models_dir = tiny_model                         # stand-in models only
    w.segmenter = pipeline.Segmenter.load_default(tiny_model)
    w.start()
    wait_idle(app, w)
    assert w.editing and w.pages.currentIndex() == 1
    assert list(out.iterdir()) == []                  # nothing written yet

    # edit: rotate, tilt, resize, move, crop result 1:1
    w.rotate(1)
    w.tilt.setValue(100)
    w.size.setValue(80)
    w.on_moved(0.05, 0.0)
    w.set_tool("out")
    w.set_aspect("1:1")
    app.processEvents()

    # colors: subject and background keep separate values
    w.set_tool("colors")
    w.tone_sliders[0][0].setValue(40)                 # subject brightness
    w.set_tone_target("bg")
    assert w.tone_sliders[0][0].value() == 0          # background has its own values
    w.tone_sliders[2][0].setValue(-30)                # background saturation
    assert w.edit.subj_tone == (40, 0, 0) and w.edit.bg_tone == (0, 0, -30)

    # blend: edges and lens blur; the Crop tab remembers result vs main photo
    w.set_tool("blend")
    w.blend_sliders["blur"][0].setValue(60)
    w.blend_sliders["shift"][0].setValue(-20)
    assert (w.edit.blur, w.edit.shift) == (60, -20)
    w.set_tool("src")
    w.set_tool("adjust")
    w.set_tool("crop")
    assert w.tool == "src" and w.crop_pages.currentIndex() == 1

    assert not w.next_btn.isVisibleTo(w)
    w.discard()
    assert not w.editing and list(out.iterdir()) == []   # Discard: nothing saved
    assert w.next_btn.isVisibleTo(w)                  # "Start new" offered

    w.start()                                         # reuses the cut-out, no re-run
    assert w.editing and w.edit == pipeline.Edit()
    w.rotate(1)
    w.set_tool("out")
    w.set_aspect("1:1")
    w.save()
    wait_idle(app, w)
    files = list(out.iterdir())
    assert [f.name for f in files] == ["photo_bg-replaced.jpg"]
    with Image.open(files[0]) as im:
        assert im.size == (60, 60)                    # 80x60 rotated -> 60x80, cropped 1:1
    assert not w.editing and w.next_btn.isVisibleTo(w)
    w.main_card.accept_path(str(bg))                  # pick another photo
    assert not w.next_btn.isVisibleTo(w) and w.view.after is None
    w.close()
    app.processEvents()


def test_no_background_and_output_folder_rule(jpegs, tiny_model, tmp_path):
    main, bg, out = jpegs
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / "second.jpg"
    Image.open(main).save(other)
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    assert w.windowTitle().startswith("ImageMagic")

    # no folder chosen yet: the output folder follows each photo
    w.main_card.accept_path(str(main))
    assert w.out_card.path == str(main.parent)
    w.main_card.accept_path(str(other))
    assert w.out_card.path == str(other_dir)

    # background is optional: ready with just a photo, and it can be cleared again
    assert w.run_btn.isEnabled() and "Edit photo" in w.run_btn.text()
    w.bg_card.accept_path(str(bg))
    assert "Replace background" in w.run_btn.text() and w.bg_card.clear_btn.isVisibleTo(w)
    w.bg_card.clear_btn.click()
    assert w.bg_card.path == "" and "Edit photo" in w.run_btn.text()

    # a folder the user chose stays for the rest of the session
    w.out_card.accept_path(str(out))
    w.main_card.accept_path(str(main))
    assert w.out_card.path == str(out)

    # edit without a new background: portrait blur, save as _edited
    w.models_dir = tiny_model                         # stand-in models only
    w.segmenter = pipeline.Segmenter.load_default(tiny_model)
    w.start()
    wait_idle(app, w)
    assert w.editing and w.prep.keep_bg
    w.set_tool("blend")
    w.blend_sliders["blur"][0].setValue(50)
    w.save()
    wait_idle(app, w)
    assert [f.name for f in out.iterdir()] == ["photo_edited.jpg"]
    assert w.out_card.path == str(out)                 # still the chosen folder
    w.close()
    app.processEvents()


def test_effect_tabs(jpegs, tiny_model):
    main, bg, out = jpegs
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    w.main_card.accept_path(str(main))
    w.bg_card.accept_path(str(bg))
    w.models_dir = tiny_model                         # stand-in models only
    w.segmenter = pipeline.Segmenter.load_default(tiny_model)
    w.start()
    wait_idle(app, w)
    for tab in ("blend", "blur", "looks"):
        w.set_tool(tab)
        assert w.tool == tab and w.view.mode == "move"
    w.blend_sliders["shadow"][0].setValue(60)
    w.blend_sliders["falloff"][0].setValue(40)
    w.blend_sliders["match"][0].setValue(70)
    w.chip_groups["look"].buttons()[4].click()             # Film
    w.chip_groups["bokeh"].buttons()[1].click()            # Hexagon
    e = w.edit
    assert (e.shadow, e.falloff, e.match, e.look, e.bokeh) == (60, 40, 70, "film", "hex")
    w.render_now()
    w.reset_fx("looks")
    assert (w.edit.match, w.edit.look) == (0, "none") and w.edit.shadow == 60
    assert w.chip_groups["look"].buttons()[0].isChecked()
    w.reset_all()
    assert w.edit == pipeline.Edit() and w.blend_sliders["focus"][0].value() == 100
    w.discard()
    w.close()
    app.processEvents()


def test_fine_cut_out_and_depth_blur(jpegs, tiny_model):
    main, bg, out = jpegs
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    w.models_dir = tiny_model
    w.main_card.accept_path(str(main))
    w.out_card.accept_path(str(out))

    # own background + depth model: depth blur is available
    w.start()
    wait_idle(app, w)
    assert w.segmenter.path.name == "BiRefNet-tiny-test.onnx"
    assert w.fine_model.path.name == "BiRefNet-matting-test.onnx"     # always used when present
    assert w.prep.depth is not None and w.blend_sliders["depth"][0].isEnabled()
    w.blend_sliders["depth"][0].setValue(60)
    w.render_now()
    assert w.edit.depth == 60
    w.discard()

    # with a new background there is no depth: the slider is disabled
    w.bg_card.accept_path(str(bg))
    w.start()
    wait_idle(app, w)
    assert w.prep.depth is None and not w.blend_sliders["depth"][0].isEnabled()
    w.discard()
    w.close()
    app.processEvents()


def test_start_new_resets_everything_but_the_output_folder(jpegs, tiny_model):
    main, bg, out = jpegs
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    w.models_dir = tiny_model
    w.main_card.accept_path(str(main))
    w.bg_card.accept_path(str(bg))
    w.out_card.accept_path(str(out))
    w.start()
    wait_idle(app, w)
    w.blend_sliders["blur"][0].setValue(50)
    w.save()
    wait_idle(app, w)
    assert w.next_btn.isVisibleTo(w) and w.next_btn.text().strip() == "Start new"

    w.next_btn.click()
    assert w.main_card.path == "" and w.bg_card.path == ""
    assert not w.bg_card.clear_btn.isVisibleTo(w)
    assert w.cache == (None, None)
    assert w.view.before is None and w.view.after is None
    assert not w.run_btn.isEnabled() and not w.next_btn.isVisibleTo(w)
    assert not w.reveal.isVisibleTo(w) and w.bar.value() == 0
    assert w.out_card.path == str(out) and w.out_card.chosen      # folder kept

    # the next photo starts clean: default edit, same output folder
    w.main_card.accept_path(str(main))
    assert w.out_card.path == str(out)
    w.start()
    wait_idle(app, w)
    assert w.edit == pipeline.Edit() and w.prep.keep_bg and w.fine_model is not None
    w.discard()
    w.close()
    app.processEvents()
