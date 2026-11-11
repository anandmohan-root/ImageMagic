import hashlib

import pytest
from PIL import Image

from bgreplacer.pipeline import Segmenter, replace_background, save_jpeg, unique_output_path


def test_name_and_counter(tmp_path):
    main = tmp_path / "IMG_01.jpeg"
    assert unique_output_path(main, tmp_path).name == "IMG_01_bg-replaced.jpg"
    (tmp_path / "IMG_01_bg-replaced.jpg").write_bytes(b"x")
    assert unique_output_path(main, tmp_path).name == "IMG_01_bg-replaced_1.jpg"
    (tmp_path / "IMG_01_bg-replaced_1.jpg").write_bytes(b"x")
    assert unique_output_path(main, tmp_path).name == "IMG_01_bg-replaced_2.jpg"


def test_save_never_overwrites(tmp_path):
    existing = tmp_path / "a_bg-replaced.jpg"
    existing.write_bytes(b"keep me")
    out = save_jpeg(Image.new("RGB", (8, 8)), tmp_path / "a.jpg", tmp_path)
    assert out.name == "a_bg-replaced_1.jpg"
    assert existing.read_bytes() == b"keep me"


def test_save_refuses_main_path(tmp_path):
    # A main image literally named like the output, in the output folder.
    main = tmp_path / "x_bg-replaced.jpg"
    Image.new("RGB", (8, 8)).save(main)
    before = main.read_bytes()
    out = save_jpeg(Image.new("RGB", (8, 8), "red"), main, tmp_path)
    assert out != main and main.read_bytes() == before


def test_full_run_keeps_main_untouched_and_exif(tiny_model, jpegs):
    main, bg, out_dir = jpegs
    digest = hashlib.sha256(main.read_bytes()).hexdigest()
    mtime = main.stat().st_mtime_ns
    seg = Segmenter.load_default(tiny_model)

    out1 = replace_background(main, bg, out_dir, seg)
    out2 = replace_background(main, bg, out_dir, seg)

    assert out1.name == "photo_bg-replaced.jpg"
    assert out2.name == "photo_bg-replaced_1.jpg"
    assert hashlib.sha256(main.read_bytes()).hexdigest() == digest
    assert main.stat().st_mtime_ns == mtime
    with Image.open(out1) as im:
        assert im.size == (80, 60)
        assert im.getexif().get(0x010F) == "TestCam"
        assert im.getpixel((2, 2))[2] > 200        # dark corner -> blue bg


def test_no_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        Segmenter.load_default(tmp_path)
