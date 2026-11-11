"""Build a portable app folder that starts through Python's own signed pythonw.exe.

    .venv\\Scripts\\python scripts\\make_portable.py

Result: dist\\ImageMagic\\
    ImageMagic.cmd            double-click to start (works wherever the folder is)
    python\\                   copy of this PC's Python runtime + only the app's libraries
    app\\                      bgreplacer package, main.py, models

Why: Smart App Control blocks our own unsigned / self-signed .exe. Here the only
programs are python.exe and pythonw.exe, signed by the Python Software Foundation.
Fully offline: everything is copied from this PC (the Python install and .venv).
"""
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "ImageMagic"
BASE = Path(sys.base_prefix)                       # the installed Python, not the venv
SITE = OUT / "python" / "Lib" / "site-packages"
# Runtime libraries only. PySide6_Essentials has QtCore/Gui/Widgets/Svg; the
# much larger PySide6_Addons (WebEngine, 3D, ...) is not needed.
ROOTS = ["PySide6_Essentials", "onnxruntime", "pillow", "opencv-python-headless", "numpy"]
SKIP_STDLIB = {"test", "idlelib", "tkinter", "turtledemo", "ensurepip", "site-packages",
               "__pycache__", "lib2to3", "venv"}
SKIP_DLLS = ("_tkinter", "tcl", "tk")


def closure(roots):
    seen, todo = {}, list(roots)
    while todo:
        dist = md.distribution(todo.pop())
        key = dist.metadata["Name"].lower().replace("_", "-")
        if key in seen:
            continue
        seen[key] = dist
        for r in dist.requires or []:
            req = Requirement(r)
            if not req.marker or req.marker.evaluate({"extra": ""}):
                todo.append(req.name)
    return seen.values()


def copy_runtime():
    py = OUT / "python"
    py.mkdir(parents=True)
    for f in BASE.iterdir():
        if f.is_file() and (f.name.lower() in ("python.exe", "pythonw.exe", "license.txt")
                            or f.suffix.lower() == ".dll"):
            shutil.copy2(f, py / f.name)
    shutil.copytree(BASE / "DLLs", py / "DLLs",
                    ignore=lambda d, names: [n for n in names if n.lower().startswith(SKIP_DLLS)])
    shutil.copytree(BASE / "Lib", py / "Lib",
                    ignore=lambda d, names: [n for n in names if n in SKIP_STDLIB])
    ver = f"python{sys.version_info.major}{sys.version_info.minor}"
    # ._pth isolates the runtime: ignores PYTHONPATH, registry and user site-packages
    (py / f"{ver}._pth").write_text("Lib\nDLLs\nLib\\site-packages\n..\\app\nimport site\n")


def copy_libraries():
    SITE.mkdir(parents=True)
    names = []
    for dist in closure(ROOTS):
        names.append(f"{dist.metadata['Name']} {dist.version}")
        for f in dist.files or []:
            rel = Path(str(f))
            if rel.parts[0] == ".." or rel.suffix.lower() == ".exe" or "__pycache__" in rel.parts:
                continue                       # console-script launchers and Qt tools: unsigned, unused
            src = Path(dist.locate_file(f))
            if src.is_file():
                dst = SITE / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    return sorted(names)


def copy_app():
    app = OUT / "app"
    shutil.copytree(ROOT / "bgreplacer", app / "bgreplacer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "main.py", app / "main.py")
    shutil.copy2(ROOT / "assets" / "app.ico", app / "app.ico")      # desktop-shortcut icon
    shutil.copytree(ROOT / "models", app / "models",
                    ignore=shutil.ignore_patterns("*.part"))
    (OUT / "ImageMagic.cmd").write_text(
        '@echo off\r\nstart "" "%~dp0python\\pythonw.exe" "%~dp0app\\main.py"\r\n')
    (OUT / "README.txt").write_text(
        "ImageMagic (offline photo editor)\r\n\r\n"
        "Double-click 'ImageMagic.cmd' to start.\r\n"
        "The whole folder can be copied to another Windows PC; no install needed.\r\n"
        "Only programs inside: python\\python.exe and python\\pythonw.exe, signed by the\r\n"
        "Python Software Foundation. Nothing connects to the internet.\r\n")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    copy_runtime()
    libs = copy_libraries()
    copy_app()
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print("Libraries:", ", ".join(libs))
    print(f"Built {OUT}  ({size / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
