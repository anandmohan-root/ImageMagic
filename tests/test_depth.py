"""Fine-hair model choice and depth-aware blur."""
import numpy as np
from PIL import Image

from bgreplacer import pipeline as P
from bgreplacer.pipeline import DepthEstimator, Edit, Prepared, background_depth


def test_model_choice(tiny_model, tmp_path):
    assert [m.name for m in P.find_models(tiny_model)] == ["BiRefNet-tiny-test.onnx"]
    assert P.find_fine_model(tiny_model).name == "BiRefNet-matting-test.onnx"
    assert P.find_fine_model(tmp_path) is None
    assert DepthEstimator.find(tiny_model).name == "depth-test.onnx"
    assert DepthEstimator.find(tmp_path) is None


def test_fuse_masks_takes_soft_edges_but_keeps_the_subject():
    general = np.zeros((200, 200), np.float32)
    general[40:160, 40:160] = 1.0
    # matting agrees on the top half (with a soft edge) but drops the bottom half
    matting = np.zeros((200, 200), np.float32)
    matting[40:100, 40:160] = 1.0
    matting[30:40, 40:160] = 0.5                              # soft "hair" above the outline
    fused = P.fuse_masks(general, matting)
    assert 0.3 < fused[35, 100] < 0.7                         # soft edge adopted
    assert fused[70, 100] == 1.0                              # core untouched
    assert fused[150, 100] > 0.95 and fused[158, 100] > 0.8   # dropped part kept from general
    assert fused[5, 5] == 0.0
    assert np.array_equal(P.fuse_masks(general, general), general)


def test_depth_estimator_output(tiny_model):
    est = DepthEstimator(DepthEstimator.find(tiny_model))
    arr = np.zeros((300, 600, 3), np.uint8)
    arr[:, 300:] = 255                                                # right half "near"
    d = est.predict(Image.fromarray(arr))
    assert d.shape[0] % 14 == 0 and d.shape[1] % 14 == 0 and d.shape[0] == 518
    assert d.min() == 0.0 and abs(d.max() - 1.0) < 1e-5
    assert d[:, -5:].mean() > 0.9 and d[:, :5].mean() < 0.1


def test_background_depth_fills_the_subject():
    depth = np.tile(np.linspace(0.1, 0.5, 100, dtype=np.float32), (80, 1))
    depth[20:60, 40:60] = 1.0                                         # subject is nearest
    alpha = np.zeros((160, 200), np.float32)
    alpha[40:120, 80:120] = 1.0
    filled, focus = background_depth(depth, alpha)
    assert abs(focus - 1.0) < 1e-6
    assert filled[40, 50] < 0.6                                       # no longer "near" there
    assert abs(filled[5, 5] - depth[5, 5]) < 0.03                     # rest unchanged


def depth_prep():
    """Own background: textured photo; left half at the subject's depth, right half far."""
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (80, 160, 3)).astype(np.uint8)
    alpha = np.zeros((80, 160), np.float32)
    alpha[30:50, 20:40] = 1.0
    depth = np.full((40, 80), 0.9, np.float32)
    depth[:, 40:] = 0.1
    img = Image.fromarray(arr)
    return Prepared(img, alpha, img, fg=img, keep_bg=True, depth=depth, depth_focus=0.9), arr


def test_depth_blur_keeps_the_subjects_distance_sharp():
    p, arr = depth_prep()
    out = np.asarray(p.render(Edit(depth=100))).astype(float)
    assert np.abs(out[:, :60] - arr[:, :60]).mean() < 1.0             # near half untouched
    assert out[:, 100:].std() < 0.6 * arr[:, 100:].astype(float).std()   # far half blurred


def test_depth_blur_follows_rotation_and_needs_depth():
    p, arr = depth_prep()
    out = np.asarray(p.render(Edit(depth=100).rotated(1))).astype(float)
    assert out.shape[:2] == (160, 80)
    assert out[100:].std() < out[:60].std()                           # far half is now at the bottom
    p.depth = None
    assert np.array_equal(np.asarray(p.render(Edit(depth=100))), np.asarray(p.render(Edit())))


def test_prepare_measures_depth_only_for_own_background(tiny_model, jpegs):
    main, bg, _ = jpegs
    seg = P.Segmenter.load_default(tiny_model)
    est = DepthEstimator(DepthEstimator.find(tiny_model))
    keep = P.prepare(main, None, seg, depth_model=est)
    assert keep.keep_bg and keep.depth is not None and 0.0 <= keep.depth_focus <= 1.0
    assert keep.preview(40).depth is keep.depth
    assert P.prepare(main, bg, seg, depth_model=est).depth is None
    assert P.prepare(main, None, seg).depth is None
    fine = P.Segmenter(P.find_fine_model(tiny_model))
    assert P.prepare(main, bg, seg, fine_model=fine).alpha.shape == keep.alpha.shape
