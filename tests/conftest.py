import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_model(tmp_path):
    """A tiny ONNX 'BiRefNet' stand-in: mean over RGB channels, 64x64 input."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 64, 64])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=1)
    graph = helper.make_graph([node], "tiny", [x], [y], initializer=[axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    d = tmp_path / "models"
    d.mkdir()
    onnx.save(model, d / "BiRefNet-tiny-test.onnx")
    onnx.save(model, d / "BiRefNet-matting-test.onnx")       # stand-in "fine hair" model

    # stand-in depth model: any image size, brighter = nearer
    x = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, "h", "w"])
    y = helper.make_tensor_value_info("predicted_depth", TensorProto.FLOAT, [1, "h", "w"])
    node = helper.make_node("ReduceMean", ["pixel_values", "axes"], ["predicted_depth"],
                            keepdims=0)
    graph = helper.make_graph([node], "tiny-depth", [x], [y], initializer=[axes])
    depth = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    depth.ir_version = 8
    onnx.save(depth, d / "depth-test.onnx")
    return d


@pytest.fixture
def jpegs(tmp_path):
    main = tmp_path / "in" / "photo.jpg"
    bg = tmp_path / "in" / "beach.jpg"
    main.parent.mkdir()
    arr = np.zeros((60, 80, 3), np.uint8)
    arr[15:45, 20:60] = 230                       # bright "subject"
    exif = Image.Exif()
    exif[0x010F] = "TestCam"                      # Make
    Image.fromarray(arr).save(main, quality=95, exif=exif.tobytes())
    Image.new("RGB", (200, 100), (0, 0, 255)).save(bg, quality=95)
    out = tmp_path / "out"
    out.mkdir()
    return main, bg, out
