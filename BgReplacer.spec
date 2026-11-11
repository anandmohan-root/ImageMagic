# PyInstaller folder build:  pyinstaller --noconfirm BgReplacer.spec
from pathlib import Path

models = [(str(p), "models") for p in Path("models").glob("*.onnx")]
models += [(str(p), "models") for p in Path("models").glob("*.txt")]

a = Analysis(
    ["main.py"],
    datas=models,
    excludes=["tkinter", "pytest", "onnx"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BgReplacer",
          console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="BgReplacer")
