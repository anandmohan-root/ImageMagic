"""One-time setup: download the segmentation models into ./models.

This is the ONLY code in the project that touches the network. It is not part
of the app package and is not bundled into the build. Run it once, then the
app works fully offline.

    python scripts/fetch_models.py

Licences: BiRefNet (general and matting) - MIT (github.com/ZhengPeng7/BiRefNet)
          ISNet / DIS - Apache-2.0 (github.com/xuebinqin/DIS)
          Depth Anything V2 Small - Apache-2.0 (github.com/DepthAnything/Depth-Anything-V2)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "BiRefNet-general-epoch_244.onnx":
        "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.onnx",
    "isnet-general-use.onnx":
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    # optional: "Fine hair" cut-out (MIT repository, same family as the main model)
    "BiRefNet-matting-epoch_100.onnx":
        "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-matting-epoch_100.onnx",
    # optional: depth-aware blur (Depth Anything V2 Small, Apache-2.0)
    "depth-anything-v2-small.onnx":
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    sums = []
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"{name}: already present, skipping")
        else:
            part = dest.with_suffix(".part")
            print(f"Downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, part)
            part.rename(dest)
        sums.append(f"{sha256(dest)}  {name}")
        print(sums[-1], flush=True)
    (MODELS_DIR / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    sys.exit(main())
