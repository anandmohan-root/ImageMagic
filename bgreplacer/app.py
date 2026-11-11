"""PySide6 UI: pick cards (click or drop), one action button, progress,
and a before/after compare view. Follows the Windows light/dark setting."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import (QByteArray, QEasingCurve, QObject, QPointF, QPropertyAnimation,
                            QRectF, QSize, Qt, QThread, QTimer, QUrl, Property, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontMetrics, QGuiApplication,
                           QImage, QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QSlider,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import pipeline

APP_NAME = "ImageMagic"
START_TEXT = "Pick a photo to start. A new background is optional."
JPEG_FILTER = "JPEG images (*.jpg *.jpeg *.JPG *.JPEG)"
JPEG_EXT = {".jpg", ".jpeg"}

# ------------------------------------------------------------------ theme

LIGHT = dict(bg="#F3F4F8", surface="#FFFFFF", surface2="#EEF0F5", border="#E2E5EC",
             text="#12141A", muted="#687082", accent="#5B5BF6", accent2="#9B5CF6",
             accent_soft="#ECECFE", success="#159947", danger="#D93636", canvas="#E9EBF1")
DARK = dict(bg="#0E1015", surface="#171A21", surface2="#1F232C", border="#2A2F3A",
            text="#ECEEF3", muted="#98A1B3", accent="#7476FF", accent2="#A77BFF",
            accent_soft="#262A45", success="#3CCB7F", danger="#FF6B6B", canvas="#12151B")


def is_dark() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return False


def stylesheet(c: dict) -> str:
    return f"""
    QWidget#root {{ background: {c['bg']}; }}
    QWidget {{ color: {c['text']}; font-family: 'Segoe UI Variable Text', 'Segoe UI'; }}
    QLabel#title {{ font-size: 22px; font-weight: 700; }}
    QLabel#badge {{ color: {c['muted']}; font-size: 12px; }}
    QLabel#section {{ color: {c['muted']}; font-size: 11px; font-weight: 600;
                      letter-spacing: 1px; }}
    QLabel#status {{ color: {c['muted']}; font-size: 12px; }}
    QLabel#status[state="ok"] {{ color: {c['success']}; font-weight: 600; }}
    QLabel#status[state="error"] {{ color: {c['danger']}; font-weight: 600; }}
    QFrame#card {{ background: {c['surface']}; border: 1px solid {c['border']};
                   border-radius: 16px; }}
    QFrame#card:hover {{ border-color: {c['accent']}; }}
    QFrame#card:focus {{ border: 2px solid {c['accent']}; }}
    QFrame#card[drop="true"] {{ border: 2px dashed {c['accent']};
                                background: {c['accent_soft']}; }}
    QLabel#cardTitle {{ font-size: 14px; font-weight: 600; background: transparent; }}
    QLabel#cardSub {{ color: {c['muted']}; font-size: 12px; background: transparent; }}
    QPushButton#primary {{
        border: none; border-radius: 14px; min-height: 52px; font-size: 15px;
        font-weight: 600; color: white;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 {c['accent']}, stop:1 {c['accent2']}); }}
    QPushButton#primary:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 {c['accent2']}, stop:1 {c['accent']}); }}
    QPushButton#primary:disabled {{ background: {c['surface2']}; color: {c['muted']}; }}
    QPushButton#clear {{ border: none; border-radius: 14px; background: {c['surface2']};
        color: {c['muted']}; font-size: 12px; font-weight: 700; }}
    QPushButton#clear:hover {{ background: {c['danger']}; color: white; }}
    QPushButton#link {{ border: none; background: transparent; color: {c['accent']};
                        font-size: 12px; font-weight: 600; padding: 0; }}
    QPushButton#link:hover {{ text-decoration: underline; }}
    QPushButton#secondary {{ border: 1px solid {c['border']}; border-radius: 14px;
        min-height: 52px; font-size: 15px; font-weight: 600; background: {c['surface']};
        color: {c['danger']}; }}
    QPushButton#secondary:hover {{ border-color: {c['danger']}; }}
    QPushButton#tool {{ border: 1px solid {c['border']}; border-radius: 10px; min-height: 38px;
        padding: 0 12px; font-size: 13px; font-weight: 600; background: {c['surface']}; }}
    QPushButton#tool:hover {{ border-color: {c['accent']}; }}
    QPushButton#tool:pressed {{ background: {c['accent_soft']}; }}
    QFrame#segment {{ background: {c['surface2']}; border-radius: 12px; }}
    QPushButton#seg {{ border: none; border-radius: 9px; min-height: 34px; font-size: 12px;
        padding: 0;
        font-weight: 600; color: {c['muted']}; background: transparent; }}
    QPushButton#seg:checked {{ background: {c['surface']}; color: {c['text']}; }}
    QPushButton#chip {{ border: 1px solid {c['border']}; border-radius: 16px; min-height: 32px;
        font-size: 13px; font-weight: 600; background: {c['surface']}; }}
    QPushButton#chip:hover {{ border-color: {c['accent']}; }}
    QPushButton#chip:checked {{ background: {c['accent']}; border-color: {c['accent']};
        color: white; }}
    QScrollArea {{ background: transparent; border: none; }}
    QWidget#page {{ background: transparent; }}
    QScrollBar:vertical {{ width: 6px; background: transparent; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 3px;
        min-height: 24px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QLabel#hint {{ color: {c['muted']}; font-size: 12px; }}
    QLabel#value {{ color: {c['text']}; font-size: 12px; font-weight: 600; }}
    QSlider:horizontal {{ min-height: 26px; }}
    QSlider::sub-page:horizontal:disabled {{ background: {c['border']}; }}
    QSlider::handle:horizontal:disabled {{ border-color: {c['border']}; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['surface2']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: white; border: 2px solid {c['accent']};
        width: 16px; height: 16px; margin: -8px 0; border-radius: 10px; }}
    QProgressBar {{ background: {c['surface2']}; border: none; border-radius: 3px;
                    max-height: 6px; min-height: 6px; }}
    QProgressBar::chunk {{ border-radius: 3px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 {c['accent']}, stop:1 {c['accent2']}); }}
    """


# ------------------------------------------------------------------ icons

ICONS = {
    "photo": '<path d="M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>'
             '<circle cx="12" cy="11" r="3"/><path d="M7 19c.8-2.6 2.8-4 5-4s4.2 1.4 5 4"/>',
    "landscape": '<path d="M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>'
                 '<path d="M3 16l5-5 4 4 3-3 6 6"/><circle cx="16" cy="9" r="1.5"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
    "lock": '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    "sparkle": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>'
               '<path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "drop": '<path d="M12 4v11"/><path d="M7 10l5 5 5-5"/><path d="M5 19h14"/>',
    "rot_left": '<path d="M4 12a8 8 0 1 0 2.6-5.9"/><path d="M4 4v4.5h4.5"/>',
    "rot_right": '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20 4v4.5h-4.5"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.6-5.9"/><path d="M4 4v4.5h4.5"/>'
             '<circle cx="12" cy="12" r="1.5"/>',
    "target": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/>'
              '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/>'
           '<circle cx="12" cy="12" r="3"/>',
    "trash": '<path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13"/>'
             '<path d="M9 7V4h6v3"/>',
}


def icon(name: str, color: str, size: int, dpr: float = 2.0) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
           f'stroke-linejoin="round">{ICONS[name]}</svg>')
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    QSvgRenderer(QByteArray(svg.encode())).render(p)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def load_preview(path: str, max_side: int) -> QPixmap:
    """Fast, EXIF-upright, downscaled preview (JPEG draft decode)."""
    with open(path, "rb") as f:
        with Image.open(f) as im:
            im.draft("RGB", (max_side, max_side))
            img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    data = img.tobytes("raw", "RGB")
    q = QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGB")
    q = QImage(img.tobytes("raw", "RGB"), img.width, img.height, 3 * img.width,
               QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def rounded(pm: QPixmap, size: int, radius: float, dpr: float = 2.0) -> QPixmap:
    """Center-crop pm to a square thumbnail with rounded corners."""
    out = QPixmap(int(size * dpr), int(size * dpr))
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, out.width(), out.height()), radius * dpr, radius * dpr)
    p.setClipPath(path)
    s = pm.scaled(out.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    p.drawPixmap((out.width() - s.width()) // 2, (out.height() - s.height()) // 2, s)
    p.end()
    out.setDevicePixelRatio(dpr)
    return out


# ------------------------------------------------------------------ worker

class Worker(QObject):
    """Runs fn(progress) on a background thread."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as e:
            self.failed.emit(str(e))


# ------------------------------------------------------------------ widgets

class PickCard(QFrame):
    """Tappable card: click / Enter / Space opens a picker; also accepts drops."""
    picked = Signal(str)
    THUMB = 52

    def __init__(self, kind: str, icon_name: str, title: str, hint: str, colors: dict,
                 optional: bool = False):
        super().__init__()
        self.kind, self.icon_name, self.hint, self.c = kind, icon_name, hint, colors
        self.path = ""
        self.chosen = False               # True once the user picked or dropped something
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.setFixedHeight(80)

        self.thumb = QLabel()
        self.thumb.setFixedSize(self.THUMB, self.THUMB)
        self.title = QLabel(title, objectName="cardTitle")
        self.sub = QLabel(hint, objectName="cardSub")
        self.sub.setMinimumWidth(10)
        self.sub.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chevron = QLabel()
        chevron.setPixmap(icon("chevron", colors["muted"], 18))
        chevron.setStyleSheet("background: transparent;")

        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addStretch()
        texts.addWidget(self.title)
        texts.addWidget(self.sub)
        texts.addStretch()
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(14)
        row.addWidget(self.thumb)
        row.addLayout(texts, 1)
        self.clear_btn = None
        if optional:
            self.clear_btn = QPushButton("✕", objectName="clear")
            self.clear_btn.setToolTip("Remove: keep the photo's own background")
            self.clear_btn.setCursor(Qt.PointingHandCursor)
            self.clear_btn.setFixedSize(28, 28)
            self.clear_btn.clicked.connect(self.clear)
            self.clear_btn.hide()
            row.addWidget(self.clear_btn)
        row.addWidget(chevron)
        self._tile()

    def _tile(self):
        """Icon tile shown until an image is chosen."""
        s, dpr = self.THUMB, 2.0
        pm = QPixmap(int(s * dpr), int(s * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.c["accent_soft"]))
        p.drawRoundedRect(QRectF(0, 0, s, s), 12, 12)
        p.drawPixmap(QPointF((s - 24) / 2, (s - 24) / 2), icon(self.icon_name, self.c["accent"], 24, dpr))
        p.end()
        self.thumb.setPixmap(pm)

    def set_path(self, path: str, preview: QPixmap | None = None):
        self.path = path
        self.sub.setToolTip(path)
        self._elide()
        if preview is not None:
            self.thumb.setPixmap(rounded(preview, self.THUMB, 12))
        elif not path:
            self._tile()
        if self.clear_btn is not None:
            self.clear_btn.setVisible(bool(path))
        self.picked.emit(path)

    def clear(self):
        self.chosen = False
        self.set_path("")

    def _elide(self):
        text = Path(self.path).name or self.path if self.path else self.hint
        if self.kind == "folder" and self.path:
            text = self.path
        fm = QFontMetrics(self.sub.font())
        self.sub.setText(fm.elidedText(text, Qt.ElideMiddle, max(40, self.sub.width())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def open_picker(self):
        if self.kind == "folder":
            path = QFileDialog.getExistingDirectory(self, self.title.text(), self.path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, self.title.text(),
                                                  str(Path(self.path).parent) if self.path else "",
                                                  JPEG_FILTER)
        if path:
            self.accept_path(path)

    def accept_path(self, path: str):
        self.chosen = True
        if self.kind == "folder":
            self.set_path(path)
        else:
            try:
                self.set_path(path, load_preview(path, 400))
            except Exception as e:
                QMessageBox.warning(self, "Can't open image", str(e))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.open_picker()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.open_picker()
        else:
            super().keyPressEvent(e)

    # drag & drop
    def _drop_path(self, mime) -> str | None:
        urls = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
        if len(urls) != 1:
            return None
        p = Path(urls[0])
        ok = p.is_dir() if self.kind == "folder" else p.suffix.lower() in JPEG_EXT
        return str(p) if ok else None

    def _set_drop(self, on: bool):
        self.setProperty("drop", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if self._drop_path(e.mimeData()):
            e.acceptProposedAction()
            self._set_drop(True)

    def dragLeaveEvent(self, e):
        self._set_drop(False)

    def dropEvent(self, e):
        self._set_drop(False)
        path = self._drop_path(e.mimeData())
        if path:
            self.accept_path(path)


class Canvas(QWidget):
    """The big preview. Modes:
    view  - photo, or before/after compare with a draggable divider
    move  - edited result; drag moves the subject, wheel resizes it
    crop  - an image with a crop rectangle (drag corners, edges or inside)
    """
    dropped = Signal(str)
    moved = Signal(float, float)       # drag delta as a fraction of the shown image
    wheeled = Signal(float)            # wheel steps (+ = bigger)
    crop_changed = Signal(tuple)       # (l, t, r, b) fractions of the shown image
    HANDLE = 14

    def __init__(self, colors: dict):
        super().__init__()
        self.c = colors
        self.before: QPixmap | None = None
        self.after: QPixmap | None = None
        self.image: QPixmap | None = None
        self.mode = "view"
        self.crop = QRectF(0, 0, 1, 1)
        self.aspect: float | None = None
        self.peek = False
        self._split = 1.0
        self.busy = False
        self.busy_text = ""
        self._angle = 0
        self._grab = None                # active crop handle
        self._last = QPointF()
        self.spin = QTimer(self, interval=16, timeout=self._tick)
        self.anim = QPropertyAnimation(self, b"split", self)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # animatable divider position, 0..1
    def get_split(self):
        return self._split

    def set_split(self, v):
        self._split = max(0.0, min(1.0, v))
        self.update()

    split = Property(float, get_split, set_split)

    # state setters
    def set_before(self, pm: QPixmap | None):
        self.mode = "view"
        self.before, self.after, self._split = pm, None, 1.0
        self.update()

    def set_after(self, pm: QPixmap):
        self.after = pm
        self.anim.stop()
        if self._same_aspect():
            self.anim.setDuration(900)
            self.anim.setStartValue(1.0)
            self.anim.setEndValue(0.5)
            self.anim.setEasingCurve(QEasingCurve.OutCubic)
            self.anim.start()
        self.update()

    def set_image(self, pm: QPixmap, mode: str):
        self.image, self.mode = pm, mode
        self.update()

    def set_crop(self, rect: tuple, aspect: float | None):
        l, t, r, b = rect
        self.crop, self.aspect = QRectF(l, t, r - l, b - t), aspect
        self.update()

    def set_busy(self, on: bool, text: str = ""):
        self.busy, self.busy_text = on, text
        self.spin.start() if on else self.spin.stop()
        self.update()

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def _same_aspect(self) -> bool:
        b, a = self.before, self.after
        return bool(b and a) and abs(b.width() / b.height() - a.width() / a.height()) < 0.01

    def shown(self) -> QPixmap | None:
        if self.mode == "view":
            return self.after if (self.after and not self._same_aspect()) else self.before
        return self.before if self.peek else self.image

    def _fit(self, pm: QPixmap | None) -> QRectF:
        area = QRectF(self.rect()).adjusted(16, 16, -16, -16)
        if pm is None:
            return area
        w, h = pm.width() / pm.devicePixelRatio(), pm.height() / pm.devicePixelRatio()
        s = min(area.width() / w, area.height() / h)
        r = QRectF(0, 0, w * s, h * s)
        r.moveCenter(area.center())
        return r

    def _crop_px(self) -> QRectF:
        r, c = self._fit(self.shown()), self.crop
        return QRectF(r.left() + c.left() * r.width(), r.top() + c.top() * r.height(),
                      c.width() * r.width(), c.height() * r.height())

    # painting
    def paintEvent(self, _):
        c = self.c
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(c["border"]), 1))
        p.setBrush(QColor(c["canvas"]))
        p.drawRoundedRect(frame, 20, 20)

        pm = self.shown()
        if pm is None:
            self._empty(p)
            return
        r = self._fit(pm)
        clip = QPainterPath()
        clip.addRoundedRect(r, 14, 14)
        p.save()
        p.setClipPath(clip)
        p.drawPixmap(r, pm, QRectF(pm.rect()))
        compare = self.mode == "view" and self._same_aspect()
        if compare:
            x = r.left() + r.width() * self._split
            p.setClipRect(QRectF(x, r.top(), r.right() - x, r.height()), Qt.IntersectClip)
            p.drawPixmap(r, self.after, QRectF(self.after.rect()))
        p.restore()

        if compare:
            self._chip(p, "Before", r.topLeft() + QPointF(12, 12), left=True)
            self._chip(p, "After", r.topRight() + QPointF(-12, 12), left=False)
            self._handle(p, r)
        elif self.mode == "view" and self.after is not None:
            self._chip(p, "Saved result", r.topLeft() + QPointF(12, 12), left=True)
        elif self.peek:
            self._chip(p, "Original", r.topLeft() + QPointF(12, 12), left=True)
        elif self.mode == "crop":
            self._crop_overlay(p, r)
        if self.busy:
            self._busy(p, r)

    def _empty(self, p: QPainter):
        c = self.c
        center = QRectF(self.rect()).center()
        p.drawPixmap(QPointF(center.x() - 20, center.y() - 56), icon("drop", c["muted"], 40))
        f = QFont(self.font())
        f.setPointSizeF(13)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(c["text"]))
        p.drawText(QRectF(0, center.y() - 4, self.width(), 24), Qt.AlignCenter,
                   "Drop a photo here")
        f.setPointSizeF(10)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(c["muted"]))
        p.drawText(QRectF(0, center.y() + 22, self.width(), 20), Qt.AlignCenter,
                   "or choose one on the left. You can edit before saving.")

    def _chip(self, p: QPainter, text: str, anchor: QPointF, left: bool):
        f = QFont(self.font())
        f.setPointSizeF(9)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        w = QFontMetrics(f).horizontalAdvance(text) + 20
        r = QRectF(anchor.x() if left else anchor.x() - w, anchor.y(), w, 24)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 120))
        p.drawRoundedRect(r, 12, 12)
        p.setPen(QColor("white"))
        p.drawText(r, Qt.AlignCenter, text)

    def _handle(self, p: QPainter, r: QRectF):
        x = r.left() + r.width() * self._split
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        knob = QRectF(0, 0, 36, 36)
        knob.moveCenter(QPointF(x, r.center().y()))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(knob.translated(0, 2))
        p.setBrush(QColor("white"))
        p.drawEllipse(knob)
        p.setPen(QPen(QColor("#4B5563"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        cy = knob.center().y()
        for d in (-1, 1):
            tip = x + d * 9
            p.drawLine(QPointF(tip - d * 4, cy - 5), QPointF(tip, cy))
            p.drawLine(QPointF(tip, cy), QPointF(tip - d * 4, cy + 5))

    def _crop_overlay(self, p: QPainter, r: QRectF):
        cr = self._crop_px()
        shade = QPainterPath()
        shade.addRect(r)
        hole = QPainterPath()
        hole.addRect(cr)
        p.fillPath(shade.subtracted(hole), QColor(0, 0, 0, 140))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 90), 1))
        for i in (1, 2):                                  # rule of thirds
            x = cr.left() + cr.width() * i / 3
            y = cr.top() + cr.height() * i / 3
            p.drawLine(QPointF(x, cr.top()), QPointF(x, cr.bottom()))
            p.drawLine(QPointF(cr.left(), y), QPointF(cr.right(), y))
        p.setPen(QPen(QColor("white"), 1.5))
        p.drawRect(cr)
        p.setPen(QPen(QColor("white"), 4, Qt.SolidLine, Qt.SquareCap))
        L = min(18.0, cr.width() / 3, cr.height() / 3)
        for (x, y, dx, dy) in ((cr.left(), cr.top(), 1, 1), (cr.right(), cr.top(), -1, 1),
                               (cr.left(), cr.bottom(), 1, -1), (cr.right(), cr.bottom(), -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))

    def _busy(self, p: QPainter, r: QRectF):
        clip = QPainterPath()
        clip.addRoundedRect(r, 14, 14)
        p.fillPath(clip, QColor(0, 0, 0, 110))
        f = QFont(self.font())
        f.setPointSizeF(11)
        f.setWeight(QFont.DemiBold)
        card = QRectF(0, 0, max(220, QFontMetrics(f).horizontalAdvance(self.busy_text) + 48), 116)
        card.moveCenter(r.center())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 23, 215))
        p.drawRoundedRect(card, 18, 18)
        ring = QRectF(0, 0, 40, 40)
        ring.moveCenter(QPointF(card.center().x(), card.top() + 38))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 60), 4))
        p.drawEllipse(ring)
        p.setPen(QPen(QColor("white"), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(ring, -self._angle * 16, 100 * 16)
        p.setFont(f)
        p.drawText(QRectF(card.left(), ring.bottom() + 12, card.width(), 24), Qt.AlignCenter,
                   self.busy_text)

    # mouse
    def _hit(self, pos: QPointF) -> str | None:
        cr, h = self._crop_px(), self.HANDLE
        near = lambda a, b: abs(a - b) <= h
        inside_y = cr.top() - h <= pos.y() <= cr.bottom() + h
        inside_x = cr.left() - h <= pos.x() <= cr.right() + h
        edge = ""
        if inside_y and near(pos.x(), cr.left()):
            edge += "l"
        elif inside_y and near(pos.x(), cr.right()):
            edge += "r"
        if inside_x and near(pos.y(), cr.top()):
            edge += "t"
        elif inside_x and near(pos.y(), cr.bottom()):
            edge += "b"
        if self.aspect and len(edge) == 1:
            edge = ""                                  # locked ratio: corners only
        if edge:
            return edge
        return "move" if cr.contains(pos) else None

    CURSORS = {"l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor, "t": Qt.SizeVerCursor,
               "b": Qt.SizeVerCursor, "lt": Qt.SizeFDiagCursor, "rb": Qt.SizeFDiagCursor,
               "rt": Qt.SizeBDiagCursor, "lb": Qt.SizeBDiagCursor, "move": Qt.SizeAllCursor}

    def mousePressEvent(self, e):
        if self.busy or e.button() != Qt.LeftButton:
            return
        pos = e.position()
        if self.mode == "view" and self._same_aspect():
            self.anim.stop()
            r = self._fit(self.before)
            self.set_split((pos.x() - r.left()) / max(1.0, r.width()))
        elif self.mode == "move":
            self._last = pos
            self.setCursor(Qt.ClosedHandCursor)
        elif self.mode == "crop":
            self._grab, self._last, self._start = self._hit(pos), pos, self._crop_px()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.busy:
            return
        if not e.buttons() & Qt.LeftButton:
            if self.mode == "crop":
                self.setCursor(self.CURSORS.get(self._hit(pos), Qt.ArrowCursor))
            elif self.mode == "move":
                self.setCursor(Qt.OpenHandCursor)
            elif self.mode == "view" and self._same_aspect():
                self.setCursor(Qt.SplitHCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return
        if self.mode == "view" and self._same_aspect():
            r = self._fit(self.before)
            self.set_split((pos.x() - r.left()) / max(1.0, r.width()))
        elif self.mode == "move" and not self.peek:
            r = self._fit(self.image)
            d = pos - self._last
            self._last = pos
            self.moved.emit(d.x() / r.width(), d.y() / r.height())
        elif self.mode == "crop" and self._grab and not self.peek:
            self._drag_crop(pos)

    def mouseReleaseEvent(self, e):
        self._grab = None
        if self.mode == "move":
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, e):
        if self.mode == "move" and not self.busy:
            self.wheeled.emit(e.angleDelta().y() / 120)

    def _drag_crop(self, pos: QPointF):
        b = self._fit(self.image)
        s, d = self._start, pos - self._last
        x0, y0, x1, y1 = s.left(), s.top(), s.right(), s.bottom()
        g, m = self._grab, 24.0                        # m: minimum size in px
        if g == "move":
            dx = min(max(d.x(), b.left() - x0), b.right() - x1)
            dy = min(max(d.y(), b.top() - y0), b.bottom() - y1)
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        else:
            if "l" in g:
                x0 = min(max(x0 + d.x(), b.left()), x1 - m)
            if "r" in g:
                x1 = max(min(x1 + d.x(), b.right()), x0 + m)
            if "t" in g:
                y0 = min(max(y0 + d.y(), b.top()), y1 - m)
            if "b" in g:
                y1 = max(min(y1 + d.y(), b.bottom()), y0 + m)
            if self.aspect and len(g) == 2:            # keep ratio, anchor opposite corner
                w = x1 - x0
                h = w / self.aspect
                room = (y1 - b.top()) if "t" in g else (b.bottom() - y0)
                if h > room:
                    h, w = room, room * self.aspect
                    if "l" in g:
                        x0 = x1 - w
                    else:
                        x1 = x0 + w
                if "t" in g:
                    y0 = y1 - h
                else:
                    y1 = y0 + h
        self.crop_changed.emit(((x0 - b.left()) / b.width(), (y0 - b.top()) / b.height(),
                                (x1 - b.left()) / b.width(), (y1 - b.top()) / b.height()))

    # dropping a main photo onto the canvas (setup screen only)
    def dragEnterEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if (len(urls) == 1 and Path(urls[0]).suffix.lower() in JPEG_EXT
                and not self.busy and self.mode == "view"):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.dropped.emit(e.mimeData().urls()[0].toLocalFile())


# ------------------------------------------------------------------ window

TOOLS = [("adjust", "Adjust"), ("colors", "Colors"), ("blend", "Blend"), ("blur", "Blur"),
         ("looks", "Looks"), ("crop", "Crop")]
# effect sliders per tab: (Edit field, label, min, max)
FX = {
    "blend": [("feather", "EDGE SOFTNESS", 0, 100), ("shift", "EDGE SHIFT", -100, 100),
              ("wrap", "LIGHT WRAP", 0, 100), ("shadow", "SHADOW", 0, 100),
              ("shadow_dir", "SHADOW DIRECTION", -100, 100),
              ("shadow_soft", "SHADOW SOFTNESS", 0, 100)],
    "blur": [("blur", "BACKGROUND BLUR", 0, 100), ("depth", "DEPTH BLUR", 0, 100),
             ("falloff", "FOCUS FALLOFF", 0, 100),
             ("focus", "FOCUS LINE", 0, 100), ("glow", "HIGHLIGHT GLOW", 0, 100)],
    "looks": [("match", "MATCH LIGHTING", 0, 100), ("vignette", "VIGNETTE", 0, 100),
              ("grain", "GRAIN", 0, 100)],
}
FX_HINTS = {
    "shift": "Left removes a halo of the old background; right brings back lost detail.",
    "depth": "Blur grows with real distance from the subject. Works only when the photo "
             "keeps its own background.",
    "focus": "Falloff adds blur away from the focus line: 0 = top of the picture, "
             "100 = bottom.",
    "match": "Pulls the subject's brightness and colour towards the background's.",
    "grain": "Fine film grain over the whole picture, so both layers share one texture.",
}
FX_CHIPS = {   # tab -> (Edit field, section label, after which slider, options, columns)
    "blur": ("bokeh", "BOKEH SHAPE", "focus", [("round", "Round"), ("hex", "Hexagon")], 2),
    "looks": ("look", "LOOK", "match", [("none", "None"), ("warm", "Warm"), ("cool", "Cool"),
                                        ("golden", "Golden"), ("film", "Film"), ("bw", "B&&W")], 3),
}
SIGNED = {"shift", "shadow_dir"}
ASPECTS = [("Free", None), ("Original", "orig"), ("1:1", 1.0), ("4:5", 0.8),
           ("3:2", 1.5), ("16:9", 16 / 9)]


def section(text: str) -> QLabel:
    return QLabel(text, objectName="section")


def tool_button(text: str, icon_name: str | None, color: str) -> QPushButton:
    b = QPushButton(("  " + text) if icon_name else text, objectName="tool")
    if icon_name:
        b.setIcon(icon(icon_name, color, 18))
        b.setIconSize(QSize(18, 18))
    b.setCursor(Qt.PointingHandCursor)
    return b


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.c = c = DARK if is_dark() else LIGHT
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(stylesheet(c))
        self.setWindowTitle(f"{APP_NAME} (offline)")
        self.setMinimumSize(1040, 720)
        self.setFocusPolicy(Qt.ClickFocus)
        self.segmenter = None             # loaded on first use, then kept
        self.fine_model = None            # matting model, loaded when Fine hair is used
        self.depth_model = None           # loaded on first use
        self.models_dir = None            # None = the app's own models folder
        self.fine = False                 # "Fine hair" cut-out chosen
        self.cache = (None, None)         # (inputs key, Prepared) - skip re-cutting
        self.thread = None
        self.last_output = ""
        self.prep = self.pv = None        # full-size and preview Prepared
        self.edit = pipeline.Edit()
        self.tool = "adjust"
        self.tone_target = "subj"         # which layer the Colors sliders change
        self.crop_target = "out"          # Crop tab: "out" = result, "src" = main photo
        self.aspect = None
        self.editing = False
        self.render_timer = QTimer(self, singleShot=True, interval=0, timeout=self.render_now)

        # header
        title = QLabel(APP_NAME, objectName="title")
        lock = QLabel()
        lock.setPixmap(icon("lock", c["muted"], 14))
        badge = QLabel("Private · runs entirely on this PC", objectName="badge")
        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        badge_row.addWidget(lock)
        badge_row.addWidget(badge)
        badge_row.addStretch()

        self.pages = QStackedWidget()
        self.pages.addWidget(self._setup_page())
        self.pages.addWidget(self._edit_page())

        self.bar = QProgressBar(textVisible=False)
        self.bar.setRange(0, 1000)
        self.bar_anim = QPropertyAnimation(self.bar, b"value", self)
        self.bar_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.status = QLabel(START_TEXT, objectName="status")
        self.status.setWordWrap(True)
        self.reveal = QPushButton("Show in folder", objectName="link")
        self.reveal.setCursor(Qt.PointingHandCursor)
        self.reveal.clicked.connect(self.show_in_folder)
        self.reveal.hide()

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(title)
        left.addLayout(badge_row)
        left.addSpacing(10)
        left.addWidget(self.pages, 1)
        left.addWidget(self.bar)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.reveal, 0, Qt.AlignTop)
        left.addLayout(status_row)
        left_w = QWidget()
        left_w.setFixedWidth(360)
        left_w.setLayout(left)

        self.view = Canvas(c)
        self.view.dropped.connect(self.main_card.accept_path)
        self.view.moved.connect(self.on_moved)
        self.view.wheeled.connect(self.on_wheel)
        self.view.crop_changed.connect(self.on_crop)

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(28)
        root.addWidget(left_w)
        root.addWidget(self.view, 1)
        self.refresh()

    # ---------------------------------------------------------- page builders
    def _setup_page(self) -> QWidget:
        c = self.c
        self.main_card = PickCard("file", "photo", "Main photo", "Choose or drop a JPEG", c)
        self.bg_card = PickCard("file", "landscape", "New background",
                                "Optional · leave empty to keep the original", c, optional=True)
        self.out_card = PickCard("folder", "folder", "Save to", "Choose a folder", c)
        self.main_card.picked.connect(self.on_main)
        for card in (self.bg_card, self.out_card):
            card.picked.connect(self.on_other)
        for card in (self.main_card, self.bg_card, self.out_card):
            shadow = QGraphicsDropShadowEffect(card, blurRadius=24, xOffset=0, yOffset=4)
            shadow.setColor(QColor(0, 0, 0, 60 if is_dark() else 18))
            card.setGraphicsEffect(shadow)
        self.run_btn = QPushButton("  Replace background", objectName="primary")
        self.run_btn.setIcon(icon("sparkle", "#FFFFFF", 20))
        self.run_btn.setIconSize(QSize(20, 20))
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.start)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        for label, card in (("PHOTO", self.main_card), ("BACKGROUND", self.bg_card),
                            ("OUTPUT", self.out_card)):
            lay.addWidget(section(label))
            lay.addWidget(card)
            lay.addSpacing(4)
        self.cut_box = QWidget()
        cut = QVBoxLayout(self.cut_box)
        cut.setContentsMargins(0, 0, 0, 0)
        cut.setSpacing(10)
        cut.addWidget(section("CUT-OUT"))
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.cut_group = QButtonGroup(self)
        for fine, text, tip in ((False, "Standard", "Crisp edges. Best for most photos."),
                                (True, "Fine hair", "Softer, see-through edges for loose hair "
                                                    "and fur. Takes about 15 s longer.")):
            chip = QPushButton(text, objectName="chip", checkable=True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip(tip)
            chip.setProperty("fine", fine)
            chip.setChecked(not fine)
            self.cut_group.addButton(chip)
            chips.addWidget(chip)
        self.cut_group.buttonClicked.connect(lambda b: self.set_fine(b.property("fine")))
        cut.addLayout(chips)
        self.cut_box.setVisible(pipeline.find_fine_model() is not None)
        lay.addWidget(self.cut_box)
        lay.addStretch()
        self.next_btn = tool_button("Start new", "reset", self.c["accent"])
        self.next_btn.setToolTip("Clear the photo, background and settings and start again. "
                                 "The output folder stays.")
        self.next_btn.clicked.connect(self.start_new)
        self.next_btn.hide()                       # shown after Save or Discard
        lay.addWidget(self.next_btn)
        lay.addWidget(self.run_btn)
        return page

    def _edit_page(self) -> QWidget:
        c, a = self.c, self.c["accent"]
        # segmented tool switch
        seg = QFrame(objectName="segment")
        seg_lay = QHBoxLayout(seg)
        seg_lay.setContentsMargins(4, 4, 4, 4)
        seg_lay.setSpacing(2)
        self.tool_group = QButtonGroup(self)
        for key, text in TOOLS:
            b = QPushButton(text, objectName="seg", checkable=True)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty("tool", key)
            self.tool_group.addButton(b)
            seg_lay.addWidget(b)
            if key == "adjust":
                b.setChecked(True)
        self.tool_group.buttonClicked.connect(lambda b: self.set_tool(b.property("tool")))

        # adjust options
        adjust = QWidget()
        al = QVBoxLayout(adjust)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        al.addWidget(section("ROTATE PHOTO"))
        rot = QHBoxLayout()
        rl, rr = tool_button("Left", "rot_left", a), tool_button("Right", "rot_right", a)
        rl.clicked.connect(lambda: self.rotate(-1))
        rr.clicked.connect(lambda: self.rotate(1))
        rot.addWidget(rl)
        rot.addWidget(rr)
        al.addLayout(rot)
        al.addSpacing(6)
        self.tilt, self.tilt_val = self._slider(al, "TILT", -450, 450, 0, self.on_tilt)
        self.size, self.size_val = self._slider(al, "SUBJECT SIZE", 10, 400, 100, self.on_size)
        al.addSpacing(6)
        al.addWidget(section("POSITION"))
        al.addWidget(QLabel("Drag the subject on the preview. Scroll to resize.",
                            objectName="hint", wordWrap=True))
        pos = QHBoxLayout()
        center, reset = tool_button("Center", "target", a), tool_button("Reset all", "reset", a)
        center.clicked.connect(self.center_subject)
        reset.clicked.connect(self.reset_all)
        pos.addWidget(center)
        pos.addWidget(reset)
        al.addLayout(pos)
        al.addStretch()

        # crop photo options
        src = QWidget()
        sl = QVBoxLayout(src)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(section("CROP MAIN PHOTO"))
        sl.addWidget(QLabel("Trim the original photo, for example to remove an object at the "
                            "edge. Drag the corners or edges on the preview. The result frame "
                            "follows this crop.", objectName="hint", wordWrap=True))
        b = tool_button("Reset photo crop", "reset", a)
        b.clicked.connect(lambda: self.set_edit(src_crop=(0.0, 0.0, 1.0, 1.0)))
        sl.addWidget(b)
        sl.addStretch()

        # crop result options
        out = QWidget()
        ol = QVBoxLayout(out)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(8)
        ol.addWidget(section("ASPECT RATIO"))
        grid = QGridLayout()
        grid.setSpacing(8)
        self.aspect_group = QButtonGroup(self)
        for i, (name, value) in enumerate(ASPECTS):
            chip = QPushButton(name, objectName="chip", checkable=True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setProperty("aspect", name)
            self.aspect_group.addButton(chip)
            grid.addWidget(chip, i // 3, i % 3)
            if value is None:
                chip.setChecked(True)
        self.aspect_group.buttonClicked.connect(lambda b: self.set_aspect(b.property("aspect")))
        ol.addLayout(grid)
        ol.addWidget(QLabel("Drag the corners or the frame on the preview.",
                            objectName="hint", wordWrap=True))
        b = tool_button("Reset result crop", "reset", a)
        b.clicked.connect(self.reset_out_crop)
        ol.addWidget(b)
        ol.addStretch()

        # colors options
        colors = QWidget()
        cl = QVBoxLayout(colors)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        cl.addWidget(section("APPLY TO"))
        targets = QHBoxLayout()
        targets.setSpacing(8)
        self.tone_group = QButtonGroup(self)
        for key, text in (("subj", "Subject"), ("bg", "Background")):
            chip = QPushButton(text, objectName="chip", checkable=True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setProperty("target", key)
            chip.setChecked(key == "subj")
            self.tone_group.addButton(chip)
            targets.addWidget(chip)
        self.tone_group.buttonClicked.connect(lambda b: self.set_tone_target(b.property("target")))
        cl.addLayout(targets)
        cl.addSpacing(6)
        self.tone_sliders = [
            self._slider(cl, label, -100, 100, 0, lambda v, i=i: self.on_tone(i, v))
            for i, label in enumerate(("BRIGHTNESS", "CONTRAST", "SATURATION"))]
        cl.addSpacing(6)
        b = tool_button("Reset colors", "reset", a)
        b.clicked.connect(self.reset_tone)
        cl.addWidget(b)
        cl.addStretch()

        # effect pages (Blend, Blur, Looks) are generated from FX / FX_CHIPS
        self.blend_sliders = {}           # every effect slider, by Edit field
        self.chip_groups = {}             # Edit field -> QButtonGroup
        defaults = pipeline.Edit()
        fx_pages = []
        for tab, sliders in FX.items():
            page = QWidget(objectName="page")
            pl = QVBoxLayout(page)
            pl.setContentsMargins(0, 0, 8, 0)
            pl.setSpacing(8)
            for key, label, lo, hi in sliders:
                self.blend_sliders[key] = self._slider(
                    pl, label, lo, hi, getattr(defaults, key),
                    lambda v, k=key: self.set_edit(**{k: v}))
                if key in FX_HINTS:
                    pl.addWidget(QLabel(FX_HINTS[key], objectName="hint", wordWrap=True))
                chips = FX_CHIPS.get(tab)
                if chips and chips[2] == key:
                    field, title, _, options, cols = chips
                    pl.addWidget(section(title))
                    grid = QGridLayout()
                    grid.setSpacing(8)
                    group = self.chip_groups[field] = QButtonGroup(self)
                    for i, (value, text) in enumerate(options):
                        chip = QPushButton(text, objectName="chip", checkable=True)
                        chip.setCursor(Qt.PointingHandCursor)
                        chip.setProperty("value", value)
                        chip.setChecked(value == getattr(defaults, field))
                        group.addButton(chip)
                        grid.addWidget(chip, i // cols, i % cols)
                    group.buttonClicked.connect(
                        lambda btn, f=field: self.set_edit(**{f: btn.property("value")}))
                    pl.addLayout(grid)
            pl.addSpacing(6)
            b = tool_button(f"Reset {tab}", "reset", a)
            b.clicked.connect(lambda _=False, t=tab: self.reset_fx(t))
            pl.addWidget(b)
            pl.addStretch()
            area = QScrollArea()
            area.setWidget(page)
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            fx_pages.append(area)

        # crop: result or main photo
        crop = QWidget()
        kl = QVBoxLayout(crop)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.setSpacing(8)
        kl.addWidget(section("CROP WHAT"))
        which = QHBoxLayout()
        which.setSpacing(8)
        self.crop_group = QButtonGroup(self)
        for key, text in (("out", "Result"), ("src", "Main photo")):
            chip = QPushButton(text, objectName="chip", checkable=True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setProperty("target", key)
            chip.setChecked(key == "out")
            self.crop_group.addButton(chip)
            which.addWidget(chip)
        self.crop_group.buttonClicked.connect(lambda b: self.set_tool(b.property("target")))
        kl.addLayout(which)
        kl.addSpacing(6)
        self.crop_pages = QStackedWidget()
        self.crop_pages.addWidget(out)
        self.crop_pages.addWidget(src)
        kl.addWidget(self.crop_pages, 1)

        self.tool_pages = QStackedWidget()
        for w in (adjust, colors, *fx_pages, crop):
            self.tool_pages.addWidget(w)

        self.compare_btn = tool_button("Hold to see original", "eye", a)
        self.compare_btn.pressed.connect(lambda: self.peek(True))
        self.compare_btn.released.connect(lambda: self.peek(False))

        self.discard_btn = QPushButton("  Discard", objectName="secondary")
        self.discard_btn.setIcon(icon("trash", c["danger"], 18))
        self.discard_btn.setIconSize(QSize(18, 18))
        self.discard_btn.setCursor(Qt.PointingHandCursor)
        self.discard_btn.clicked.connect(self.discard)
        self.save_btn = QPushButton("  Save", objectName="primary")
        self.save_btn.setIcon(icon("drop", "#FFFFFF", 20))
        self.save_btn.setIconSize(QSize(20, 20))
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self.save)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.discard_btn, 1)
        actions.addWidget(self.save_btn, 1)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(seg)
        lay.addWidget(self.tool_pages, 1)
        lay.addWidget(self.compare_btn)
        lay.addLayout(actions)
        self.edit_page = page
        return page

    def _slider(self, lay, label, lo, hi, val, slot):
        head = QHBoxLayout()
        head.addWidget(section(label))
        value = QLabel(objectName="value")
        head.addStretch()
        head.addWidget(value)
        lay.addLayout(head)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setCursor(Qt.PointingHandCursor)
        s.valueChanged.connect(slot)
        lay.addWidget(s)
        return s, value

    # ---------------------------------------------------------- setup state
    def ready(self) -> bool:
        return bool(self.main_card.path and self.out_card.path)

    def action_name(self) -> str:
        return "Replace background" if self.bg_card.path else "Edit photo"

    def ready_text(self) -> str:
        return f"Ready. Click {self.action_name()}."

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def refresh(self, *_):
        busy = self.busy()
        self.run_btn.setEnabled(self.ready() and not busy)
        self.run_btn.setText("  " + self.action_name())
        self.edit_page.setEnabled(not busy)
        self.tilt_val.setText(f"{self.edit.tilt:+.1f}°".replace("+0.0", "0.0"))
        self.size_val.setText(f"{round(self.edit.scale * 100)}%")
        for slider, label in self.tone_sliders:
            label.setText(f"{slider.value():+d}" if slider.value() else "0")
        for key, (slider, label) in self.blend_sliders.items():
            v = slider.value()
            label.setText(f"{v:+d}" if (key in SIGNED and v) else str(v))

    def on_main(self, path: str):
        try:
            self.view.set_before(load_preview(path, 1800))
        except Exception:
            self.view.set_before(None)
        if not self.out_card.chosen:          # a folder you chose stays for this session;
            self.out_card.set_path(str(Path(path).parent))   # otherwise follow the photo
        self.reveal.hide()
        self.next_btn.hide()
        self.set_status(self.ready_text())
        self.refresh()

    def start_new(self):
        """Back to a fresh start: no photo, no background, Standard cut-out.
        Only the output folder (and whether you chose it) is kept."""
        if self.busy():
            return
        for card in (self.main_card, self.bg_card):
            card.blockSignals(True)            # plain reset: no "photo picked" handling
            card.clear()
            card.blockSignals(False)
        self.set_fine(False)
        self.cache = (None, None)              # also frees the last cut-out's memory
        self.last_output = ""
        self.view.set_before(None)
        self.bar_anim.stop()
        self.bar.setValue(0)
        self.reveal.hide()
        self.next_btn.hide()
        self.set_status(START_TEXT)
        self.refresh()

    def set_fine(self, fine: bool):
        self.fine = bool(fine)
        for chip in self.cut_group.buttons():
            chip.setChecked(chip.property("fine") == self.fine)

    def on_other(self, *_):
        if self.ready():
            self.set_status(self.ready_text())
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self.setFocus()               # no card looks "selected" on launch

    def set_status(self, text: str, state: str = ""):
        self.status.setText(text)
        self.status.setProperty("state", state)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def animate_bar(self, pct: int):
        self.bar_anim.stop()
        self.bar_anim.setDuration(450)
        self.bar_anim.setStartValue(self.bar.value())
        self.bar_anim.setEndValue(pct * 10)
        self.bar_anim.start()

    # ---------------------------------------------------------- background jobs
    FRIENDLY = {"Loading model": "Warming up…", "Loading images": "Opening photos…",
                "Segmenting": "Finding the subject…", "Refining": "Refining edges…",
                "Preparing": "Opening editor…", "Filling": "Filling in the background…", "Estimating": "Measuring depth…",
                "Tracing": "Tracing fine hair…", "Rendering": "Rendering full size…",
                "Saving": "Saving…"}

    def run_job(self, fn, on_done, busy_text: str):
        self.reveal.hide()
        self.bar.setValue(0)
        self.set_status(busy_text)
        self.view.set_busy(True, busy_text)
        self.thread = QThread()
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(self.on_failed)
        for sig in (self.worker.done, self.worker.failed):
            sig.connect(self.thread.quit)
        self.thread.finished.connect(self.refresh)
        self.thread.start()
        self.refresh()

    def on_progress(self, pct: int, msg: str):
        self.animate_bar(pct)
        self.set_status(msg)
        friendly = next((v for k, v in self.FRIENDLY.items() if msg.startswith(k)), msg)
        self.view.set_busy(True, friendly)

    def on_failed(self, msg: str):
        self.view.set_busy(False)
        self.bar.setValue(0)
        self.set_status("Something went wrong.", "error")
        QMessageBox.critical(self, "Error", msg)

    def start(self):
        """Cut out the subject (or reuse the last cut-out) and open the editor."""
        if not self.ready() or self.busy():
            return
        main, bg = self.main_card.path, (self.bg_card.path or None)
        fine = self.fine
        key = (main, Path(main).stat().st_mtime_ns, bg,
               Path(bg).stat().st_mtime_ns if bg else None, fine)
        if self.cache[0] == key:
            self.enter_edit(self.cache[1])
            return

        def job(progress):
            if self.segmenter is None:
                progress(2, "Loading model (first run only)…")
                self.segmenter = pipeline.Segmenter.load_default(self.models_dir)
            fine_path = pipeline.find_fine_model(self.models_dir) if fine else None
            if fine_path and self.fine_model is None:
                progress(3, "Loading model (fine hair)…")
                self.fine_model = pipeline.Segmenter(fine_path)
            depth_path = None if bg else pipeline.DepthEstimator.find(self.models_dir)
            if depth_path and self.depth_model is None:
                self.depth_model = pipeline.DepthEstimator(depth_path)
            prep = pipeline.prepare(main, bg, self.segmenter, progress,
                                    self.depth_model if depth_path else None,
                                    self.fine_model if fine_path else None)
            self.cache = (key, prep)
            return prep

        self.run_job(job, self.enter_edit, "Starting…")

    # ---------------------------------------------------------- editor
    def enter_edit(self, prep):
        self.prep, self.pv = prep, prep.preview(1400)
        self.blend_sliders["depth"][0].setEnabled(prep.depth is not None)
        self.editing = True
        self.edit = pipeline.Edit()
        self.aspect = None
        for slider, v in ((self.tilt, 0), (self.size, 100)):
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
        for b in self.aspect_group.buttons():
            b.setChecked(b.property("aspect") == "Free")
        self.sync_tone_sliders()
        self.sync_blend_sliders()
        self.view.set_busy(False)
        self.animate_bar(100)
        self.pages.setCurrentIndex(1)
        self.set_tool("adjust")
        self.set_status("Adjust if you like, then Save. Nothing is saved yet.")

    def set_tool(self, tool: str):
        """tool: adjust / colors / blend, or crop (last crop target), src or out."""
        if tool == "crop":
            tool = self.crop_target
        self.tool = tool
        tab = "crop" if tool in ("src", "out") else tool
        for b in self.tool_group.buttons():
            b.setChecked(b.property("tool") == tab)
        self.tool_pages.setCurrentIndex([k for k, _ in TOOLS].index(tab))
        if tab == "crop":
            self.crop_target = tool
            self.crop_pages.setCurrentIndex(0 if tool == "out" else 1)
            for b in self.crop_group.buttons():
                b.setChecked(b.property("target") == tool)
        self.render_now()

    def set_edit(self, **changes):
        self.edit = replace(self.edit, **changes)
        own_crop = {"src": "src_crop", "out": "out_crop"}.get(self.tool)
        if set(changes) == {own_crop}:         # the shown image is unchanged by its own crop
            self.view.set_crop(changes[own_crop], self.view.aspect)
        else:
            self.render_timer.start()          # coalesce rapid changes
        self.update_size_label()
        self.refresh()

    def render_now(self):
        if self.pv is None:
            return
        e = self.edit
        if self.tool not in ("src", "out"):
            self.view.set_image(pil_to_pixmap(self.pv.render(e)), "move")
        elif self.tool == "out":
            self.view.set_crop(e.out_crop, self.aspect_value())
            self.view.set_image(pil_to_pixmap(self.pv.render(e, crop=False)), "crop")
        else:
            img, _ = self.pv.photo(replace(e, src_crop=(0.0, 0.0, 1.0, 1.0)))
            self.view.set_crop(e.src_crop, None)
            self.view.set_image(pil_to_pixmap(img), "crop")
        self.update_size_label()

    def update_size_label(self):
        if self.prep is not None and self.editing:
            w, h = self.prep.output_size(self.edit)
            self.set_status(f"Output {w} × {h} px. Nothing is saved until you press Save.")

    # editor actions
    def rotate(self, k: int):
        self.edit = self.edit.rotated(k)
        self.render_now()

    def on_tilt(self, v: int):
        self.set_edit(tilt=v / 10)

    def on_size(self, v: int):
        self.set_edit(scale=v / 100)

    def on_wheel(self, steps: float):
        scale = min(4.0, max(0.1, self.edit.scale * (1.1 ** steps)))
        self.size.setValue(round(scale * 100))          # triggers on_size

    def on_moved(self, fx: float, fy: float):
        l, t, r, b = self.edit.out_crop
        ox, oy = self.edit.offset
        self.set_edit(offset=(ox + fx * (r - l), oy + fy * (b - t)))

    def on_crop(self, rect: tuple):
        self.set_edit(**{("src_crop" if self.tool == "src" else "out_crop"): tuple(rect)})

    def center_subject(self):
        self.set_edit(offset=self.prep.centering_offset(self.edit))

    def reset_all(self):
        self.edit = pipeline.Edit()
        for slider, v in ((self.tilt, 0), (self.size, 100)):
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
        self.aspect = None
        for b in self.aspect_group.buttons():
            b.setChecked(b.property("aspect") == "Free")
        self.sync_tone_sliders()
        self.sync_blend_sliders()
        self.render_now()
        self.refresh()

    # colors
    def tone_field(self) -> str:
        return "subj_tone" if self.tone_target == "subj" else "bg_tone"

    def set_tone_target(self, target: str):
        self.tone_target = target
        self.sync_tone_sliders()

    def sync_tone_sliders(self):
        for (slider, _), v in zip(self.tone_sliders, getattr(self.edit, self.tone_field())):
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
        self.refresh()

    def on_tone(self, i: int, v: int):
        values = list(getattr(self.edit, self.tone_field()))
        values[i] = v
        self.set_edit(**{self.tone_field(): tuple(values)})

    # blend
    def sync_blend_sliders(self):
        for key, (slider, _) in self.blend_sliders.items():
            slider.blockSignals(True)
            slider.setValue(getattr(self.edit, key))
            slider.blockSignals(False)
        for field, group in self.chip_groups.items():
            for chip in group.buttons():
                chip.setChecked(chip.property("value") == getattr(self.edit, field))
        self.refresh()

    def reset_fx(self, tab: str):
        """Put one effect tab (blend / blur / looks) back to its defaults."""
        fields = [key for key, *_ in FX[tab]]
        if tab in FX_CHIPS:
            fields.append(FX_CHIPS[tab][0])
        defaults = pipeline.Edit()
        self.set_edit(**{f: getattr(defaults, f) for f in fields})
        self.sync_blend_sliders()

    def reset_tone(self):
        self.set_edit(subj_tone=(0, 0, 0), bg_tone=(0, 0, 0))
        self.sync_tone_sliders()

    def aspect_value(self) -> float | None:
        if self.aspect == "orig":
            w, h = self.prep.output_size(replace(self.edit, out_crop=(0.0, 0.0, 1.0, 1.0)))
            return w / h
        return self.aspect

    def set_aspect(self, name: str):
        self.aspect = dict(ASPECTS)[name]
        ratio = self.aspect_value()
        rect = (0.0, 0.0, 1.0, 1.0)
        if ratio:
            w, h = self.prep.output_size(replace(self.edit, out_crop=(0.0, 0.0, 1.0, 1.0)))
            frame = w / h
            cw, ch = (ratio / frame, 1.0) if frame > ratio else (1.0, frame / ratio)
            rect = ((1 - cw) / 2, (1 - ch) / 2, (1 + cw) / 2, (1 + ch) / 2)
        self.edit = replace(self.edit, out_crop=rect)
        self.render_now()

    def reset_out_crop(self):
        for b in self.aspect_group.buttons():
            b.setChecked(b.property("aspect") == "Free")
        self.set_aspect("Free")

    def peek(self, on: bool):
        self.view.peek = on
        self.view.update()

    def leave_edit(self):
        self.editing = False
        self.prep = self.pv = None
        self.pages.setCurrentIndex(0)
        self.view.set_busy(False)

    def discard(self):
        self.leave_edit()
        try:
            self.view.set_before(load_preview(self.main_card.path, 1800))
        except Exception:
            self.view.set_before(None)
        self.bar.setValue(0)
        self.next_btn.show()
        self.set_status("Discarded. Nothing was saved.")

    def save(self):
        if self.busy() or self.prep is None:
            return
        prep, edit = self.prep, self.edit
        main, out_dir = self.main_card.path, self.out_card.path

        def job(progress):
            progress(20, "Rendering full size")
            img = prep.render(edit)
            progress(85, "Saving")
            return str(pipeline.save_jpeg(img, main, out_dir, prep.exif, prep.icc, prep.suffix))

        self.run_job(job, self.on_saved, "Saving…")

    def on_saved(self, out: str):
        self.last_output = out
        self.leave_edit()
        self.animate_bar(100)
        try:
            self.view.set_before(load_preview(self.main_card.path, 1800))
            self.view.set_after(load_preview(out, 1800))
        except Exception:
            pass
        self.set_status(f"✓  Saved as {Path(out).name}", "ok")
        self.next_btn.show()
        self.reveal.show()

    def show_in_folder(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def closeEvent(self, e):
        if self.busy():
            QMessageBox.information(self, "Please wait", "Still working. Try again in a moment.")
            e.ignore()
            return
        if self.editing:
            answer = QMessageBox.question(self, "Discard this edit?",
                                          "The edited picture has not been saved. Close anyway?")
            if answer != QMessageBox.Yes:
                e.ignore()
                return
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())
