"""Print sheets: markers must be detectable with the exact IDs and nominal size; PDFs must be exactly A4."""

from __future__ import annotations

import re

import cv2
import numpy as np

from digitaltrack.printsheets import ARUCO_DICT, box_texture_sheet, marker_disc_sheet, mm, write_sheets


def test_marker_sheet_detects_ids_0_to_5_at_60_mm():
    img = marker_disc_sheet()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(img)
    assert sorted(ids.ravel().tolist()) == [0, 1, 2, 3, 4, 5]
    for c in corners:
        side = np.linalg.norm(c[0][0] - c[0][1])
        assert abs(side - mm(60)) < 3  # pixels at 300 dpi


def test_texture_sheets_differ_by_seed_and_are_not_blank():
    a, b = box_texture_sheet(0), box_texture_sheet(1)
    assert not np.array_equal(a, b)
    assert 0.1 < (a < 128).mean() < 0.6


def test_written_pdfs_are_a4(tmp_path):
    files = write_sheets(tmp_path, texture_seeds=(0,))
    pdfs = [f for f in files if f.suffix == ".pdf"]
    assert len(pdfs) == 2
    for p in pdfs:
        data = p.read_bytes()
        assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
        box = [float(v) for v in re.search(rb"/MediaBox \[([^\]]+)\]", data).group(1).split()]
        assert abs(box[2] / 72 * 25.4 - 210) < 0.1 and abs(box[3] / 72 * 25.4 - 297) < 0.1
    png = cv2.imread(str(tmp_path / "marker_disc.png"), cv2.IMREAD_GRAYSCALE)
    assert png.shape == (mm(297), mm(210))
