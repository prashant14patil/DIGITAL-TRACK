"""Print-ready A4 sheets for capture sessions (PNG at 300 dpi + PDF at exact A4 size).

* marker_disc: 6 ArUco markers (DICT_4X4_50, IDs 0-5), 60 mm each, for an OPTIONAL hand-turned disc (pose reference).
* box_texture: a non-repeating random pattern for the faces of the test box (gives registration something to lock on).

Each sheet carries a 100 mm scale bar. Printers often rescale: print at "Actual size / 100 %", MEASURE the bar and the
printed marker side with a ruler, and record the measured values; the nominal sizes are not trusted.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np

DPI = 300
A4_MM = (210.0, 297.0)
ARUCO_DICT = "DICT_4X4_50"


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def _page() -> np.ndarray:
    return np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)


def _text(img, s, x_mm, y_mm, scale=1.6, thick=3):
    cv2.putText(img, s, (mm(x_mm), mm(y_mm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)


def _scale_bar(img, x_mm, y_mm, length_mm=100.0):
    x0, y0 = mm(x_mm), mm(y_mm)
    cv2.rectangle(img, (x0, y0), (x0 + mm(length_mm), y0 + mm(4)), 0, -1)
    for k in range(0, int(length_mm) + 1, 10):
        cv2.line(img, (x0 + mm(k), y0 - mm(3 if k % 50 else 6)), (x0 + mm(k), y0), 0, 3)
    _text(img, f"{length_mm:.0f} mm scale bar - measure it after printing", x_mm, y_mm - 8, 1.2, 2)


def marker_disc_sheet(marker_mm: float = 60.0, ids=range(6)) -> np.ndarray:
    img = _page()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    _text(img, f"DIGITAL TRACK - marker disc ({ARUCO_DICT}, IDs 0-5, {marker_mm:.0f} mm)", 12, 12, 1.5, 3)
    _text(img, "Print at Actual size / 100 %. Cut along the dashed boxes; glue flat around the disc rim, evenly spaced.", 12, 18, 1.0, 2)
    _text(img, "Write down the MEASURED black-square side length of one printed marker.", 12, 23, 1.0, 2)
    gap = 18.0
    for k, mid in enumerate(ids):
        col, row = k % 2, k // 2
        x, y = 25 + col * (marker_mm + 2 * gap + 10), 36 + row * (marker_mm + 22)
        m = cv2.aruco.generateImageMarker(d, int(mid), mm(marker_mm))
        img[mm(y):mm(y) + m.shape[0], mm(x):mm(x) + m.shape[1]] = m
        # dashed cut box with a white quiet zone of 10 mm around the marker
        x0, y0, x1, y1 = mm(x - 10), mm(y - 10), mm(x + marker_mm + 10), mm(y + marker_mm + 10)
        for t in range(x0, x1, 40):
            cv2.line(img, (t, y0), (min(t + 20, x1), y0), 120, 2)
            cv2.line(img, (t, y1), (min(t + 20, x1), y1), 120, 2)
        for t in range(y0, y1, 40):
            cv2.line(img, (x0, t), (x0, min(t + 20, y1)), 120, 2)
            cv2.line(img, (x1, t), (x1, min(t + 20, y1)), 120, 2)
        _text(img, f"ID {mid}", x, y + marker_mm + 15, 1.2, 2)  # below the 10 mm white quiet zone
    _scale_bar(img, 22, 288)
    return img


def box_texture_sheet(seed: int = 0) -> np.ndarray:
    """Random non-repeating, separate blobs (radius 3-10 mm, ~35 % ink) clipped to the frame, for the box faces."""
    img = _page()
    rng = np.random.default_rng(seed)
    _text(img, f"DIGITAL TRACK - box texture (seed {seed}). Print several (different seeds); cover each box face.", 12, 14, 1.1, 2)
    x0, y0, x1, y1 = mm(10), mm(22), mm(200), mm(272)
    tile = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    th, tw = tile.shape
    while (tile < 128).mean() < 0.35:
        cx, cy = int(rng.integers(0, tw)), int(rng.integers(0, th))
        ax, ay = mm(rng.uniform(3, 10)), mm(rng.uniform(3, 10))
        if rng.random() < 0.65:
            cv2.ellipse(tile, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 0, -1, cv2.LINE_AA)
        else:
            ang = np.sort(rng.uniform(0, 2 * np.pi, 5))
            pts = np.stack([cx + ax * np.cos(ang), cy + ay * np.sin(ang)], 1).astype(np.int32)
            cv2.fillConvexPoly(tile, pts, 0, cv2.LINE_AA)
    img[y0:y1, x0:x1] = tile  # clipped to the frame: nothing covers the title or the scale bar
    cv2.rectangle(img, (x0, y0), (x1, y1), 160, 2)
    _scale_bar(img, 22, 288)
    return img


def _pdf_with_image(img: np.ndarray, path: Path) -> None:
    """Minimal one-page A4 PDF embedding the greyscale page losslessly (Flate), scaled to exactly 210 x 297 mm."""
    h, w = img.shape
    data = zlib.compress(np.ascontiguousarray(img).tobytes(), 9)
    pw, ph = 595.276, 841.890  # A4 in points
    content = f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3f} {ph:.3f}] /Resources << /XObject << /Im0 4 0 R >> >> "
        f"/Contents 5 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray /BitsPerComponent 8 "
        f"/Filter /FlateDecode /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def write_sheets(out_dir: str | Path, texture_seeds=(0, 1, 2)) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages = [("marker_disc", marker_disc_sheet())] + [(f"box_texture_{s}", box_texture_sheet(s)) for s in texture_seeds]
    written = []
    for name, img in pages:
        png, pdf = out / f"{name}.png", out / f"{name}.pdf"
        ok, buf = cv2.imencode(".png", img)
        png.write_bytes(buf.tobytes())
        _pdf_with_image(img, pdf)
        written += [png, pdf]
    return written
