# -*- coding: utf-8 -*-
"""
Analyze student survey PDFs (checkbox + handwritten text) into one Excel workbook.

Features
--------
- Processes all PDFs from a given folder.
- Each PDF -> separate worksheet in one Excel workbook.
- Each page -> one row.
- Columns: page number, A1.1–A1.5, A2.1–A2.5, A3.1–A3.4, A4.1–A4.6, B1–B5 (OCR text).
- Last row contains averages of closed questions (ignoring value 6 = 'no answer').
- Automatic checkbox detection (no manual coordinates) and validation:
  * if zero or more than one marked in a question row -> NaN
- OCR improved for handwritten / printed capital letters.

Usage
-----
python analyze_surveys.py --pdf-dir ./pdf --out ./wyniki_ankiet.xlsx

Notes
-----
- Requires: pdf2image (Poppler), pytesseract (Tesseract OCR), OpenCV, Pillow, pandas, openpyxl.
- For Windows, install Poppler and Tesseract and add to PATH.

Author: M365 Copilot
"""
from __future__ import annotations
import os
import re
import sys
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ----------------------------
# Configuration
# ----------------------------
CLOSED_QUESTIONS = (
    [f"A1.{i}" for i in range(1, 6)] +
    [f"A2.{i}" for i in range(1, 6)] +
    [f"A3.{i}" for i in range(1, 4+1)] +
    [f"A4.{i}" for i in range(1, 6+1)]
)
OPEN_QUESTIONS = [f"B{i}" for i in range(1, 6)]

# parameters that may be tuned per template
TOP_CLOSED_PART_RATIO = 0.60  # top part of page considered for closed questions
ROW_Y_TOL_FACTOR = 0.6        # row grouping tolerance as fraction of avg checkbox height
CHECKBOX_MIN = 14             # min checkbox size (px)
CHECKBOX_MAX = 55             # max checkbox size (px)
CHECKBOX_ASPECT_TOL = 0.35    # allowed aspect anomaly |1 - w/h| < tol
FILL_MIN_THRESHOLD = 0.12     # minimal fill ratio to accept any mark
MULTI_DELTA = 0.06            # if >1 boxes within (max - MULTI_DELTA) -> multi-mark
EXPECTED_ROWS = 5 + 5 + 4 + 6 # 20 question rows
EXPECTED_BOXES_PER_ROW = 6

# ----------------------------
# Utilities
# ----------------------------

def average_without_six(values: list[float|int|None]) -> float|None:
    """Average excluding 6 and NaN. Returns None if no valid values."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)) and int(v) != 6]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def to_excel_sheet_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    # Excel sheet name max 31 chars, no []:*?/\\
    safe = re.sub(r"[\[\]:\\/*?]", "_", base)[:31]
    return safe or "PDF"


# ----------------------------
# Image processing
# ----------------------------

def load_page_gray(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)


def binarize_for_boxes(gray: np.ndarray) -> np.ndarray:
    # gentle denoise + adaptive threshold inverted to make ink = white
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 7
    )
    # close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def find_checkboxes(gray: np.ndarray) -> list[tuple[int,int,int,int]]:
    th = binarize_for_boxes(gray)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if CHECKBOX_MIN <= w <= CHECKBOX_MAX and CHECKBOX_MIN <= h <= CHECKBOX_MAX:
            if abs(1.0 - (w / float(h))) < CHECKBOX_ASPECT_TOL:
                boxes.append((x, y, w, h))
    # Keep only top closed-questions part (to avoid picking text boxes from B part)
    h_img = gray.shape[0]
    y_cut = int(h_img * TOP_CLOSED_PART_RATIO)
    boxes = [b for b in boxes if b[1] + b[3]//2 <= y_cut]
    return boxes


def group_by_rows(boxes: list[tuple[int,int,int,int]]) -> list[list[tuple[int,int,int,int]]]:
    if not boxes:
        return []
    # sort by y then x
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    avg_h = np.median([h for _,_,_,h in boxes_sorted])
    tol = max(8, int(avg_h * ROW_Y_TOL_FACTOR))
    rows: list[list[tuple[int,int,int,int]]] = []
    for b in boxes_sorted:
        x, y, w, h = b
        placed = False
        for row in rows:
            # compare to first element of row by y-center
            y0 = row[0][1] + row[0][3]//2
            yc = y + h//2
            if abs(yc - y0) <= tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    # sort each row by x
    for r in rows:
        r.sort(key=lambda b: b[0])
    # keep only rows that look like questions (>=5 boxes)
    rows = [r for r in rows if len(r) >= EXPECTED_BOXES_PER_ROW - 1]
    # If a row has more than 6 boxes, try to remove outliers by x-spacing clustering
    norm_rows = []
    for r in rows:
        if len(r) == EXPECTED_BOXES_PER_ROW:
            norm_rows.append(r)
        else:
            # Heuristic: keep 6 boxes with most similar size to median
            sizes = np.array([w*h for (_,_,w,h) in r])
            med = np.median(sizes)
            idx = np.argsort(np.abs(sizes - med))[:EXPECTED_BOXES_PER_ROW]
            kept = [r[i] for i in idx]
            kept.sort(key=lambda b: b[0])
            norm_rows.append(kept)
    # sort rows top->bottom
    norm_rows.sort(key=lambda r: r[0][1])
    # keep only expected number of rows if there are more
    if len(norm_rows) >= EXPECTED_ROWS:
        norm_rows = norm_rows[:EXPECTED_ROWS]
    return norm_rows


def compute_fill_ratio(th_inv: np.ndarray, box: tuple[int,int,int,int]) -> float:
    x, y, w, h = box
    roi = th_inv[y:y+h, x:x+w]
    # In our binary_inv, ink is white (255). Fill ratio = white pixels / area
    return float(np.count_nonzero(roi)) / float(max(1, w*h))


def decide_mark(gray: np.ndarray, row: list[tuple[int,int,int,int]]) -> float | None:
    """Return selected option (1..6) or NaN if none or multiple marks.
       Validation: if >1 boxes above (max - MULTI_DELTA) and max > FILL_MIN_THRESHOLD -> NaN.
    """
    th_inv = binarize_for_boxes(gray)
    ratios = [compute_fill_ratio(th_inv, b) for b in row]
    mx = max(ratios)
    if mx < FILL_MIN_THRESHOLD:
        return math.nan  # no clear mark
    # boxes near the max
    near = [i for i, r in enumerate(ratios) if (mx - r) <= MULTI_DELTA and r >= FILL_MIN_THRESHOLD]
    if len(near) == 1:
        return float(near[0] + 1)
    else:
        return math.nan  # ambiguous (multi marks)


# ----------------------------
# OCR for open questions
# ----------------------------

def preprocess_handwriting(gray: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoising(gray, h=25)
    den = cv2.normalize(den, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.adaptiveThreshold(
        den, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 9
    )
    # Slight dilation to connect strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def extract_open_text(gray: np.ndarray) -> str:
    cfg = (
        "--oem 1 "
        "--psm 6 "
        "-l pol "
        "-c preserve_interword_spaces=1 "
        "-c tessedit_char_blacklist=|{}[]<>\n\f\r"
    )
    return pytesseract.image_to_string(gray, config=cfg)


def parse_open_answers(text: str) -> dict:
    # Normalize
    t = text
    t = t.replace("\r", "\n")
    t = re.sub(r"\u2013|\u2014|–|—", "-", t)
    # Unify markers B1..B5 (accept variants like 'B.1', 'B1', 'B 1')
    answers = {f"B{i}": "" for i in range(1,6)}
    # Build regex with positions
    pattern = re.compile(
        r"(?P<label>B\s*\.??\s*(?P<num>[1-5]))\s*[):.-]?\s*(?P<body>.*?)" \
        r"(?=(?:B\s*\.??\s*[1-5]\s*[):.-]?|\Z))",
        re.IGNORECASE | re.S
    )
    for m in pattern.finditer(t):
        num = int(m.group("num"))
        body = m.group("body").strip()
        answers[f"B{num}"] = body
    return answers


# ----------------------------
# PDF processing
# ----------------------------

def process_pdf(pdf_path: str, dpi: int = 300, debug_dir: str | None = None) -> pd.DataFrame:
    pages = convert_from_path(pdf_path, dpi=dpi)
    rows = []

    for page_no, pil_page in enumerate(pages, start=1):
        gray = load_page_gray(pil_page)
        h, w = gray.shape

        # Closed questions: detect checkboxes in top part
        boxes = find_checkboxes(gray)
        rows_boxes = group_by_rows(boxes)

        record = {"strona": page_no}

        # Map rows to questions by order
        for q_name, row_boxes in zip(CLOSED_QUESTIONS, rows_boxes):
            val = decide_mark(gray, row_boxes)
            record[q_name] = val
        # If fewer rows found than expected, fill rest with NaN
        if len(rows_boxes) < len(CLOSED_QUESTIONS):
            for q_name in CLOSED_QUESTIONS[len(rows_boxes):]:
                record[q_name] = math.nan

        # Open questions: bottom part
        y_open = int(h * TOP_CLOSED_PART_RATIO)
        open_part = gray[y_open:h, 0:w]
        open_bin = preprocess_handwriting(open_part)
        text = extract_open_text(open_bin)
        ans = parse_open_answers(text)
        for b in OPEN_QUESTIONS:
            record[b] = ans.get(b, "")

        rows.append(record)

        # Optional debug images
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # draw detected rows/boxes
            for r in rows_boxes:
                color = (0, 200, 0)
                for (x, y, bw, bh) in r:
                    cv2.rectangle(dbg, (x, y), (x+bw, y+bh), color, 2)
            cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(pdf_path)}_p{page_no:03d}.png"), dbg)

    df = pd.DataFrame(rows)

    # Append average row over closed questions (ignore 6)
    avg_row = {"strona": "ŚREDNIA"}
    for q in CLOSED_QUESTIONS:
        avg_row[q] = average_without_six(df[q].tolist() if q in df else [])
    for b in OPEN_QUESTIONS:
        avg_row[b] = ""
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    return df


# ----------------------------
# Main
# ----------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyze survey PDFs into Excel workbook")
    ap.add_argument("--pdf-dir", default="./ankiety_2025_26_zima_sem_1", help="Folder with PDF files")
    ap.add_argument("--out", default="wyniki_ankiet.xlsx", help="Output Excel path")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI for PDF pages")
    ap.add_argument("--debug-dir", default=None, help="Optional folder to save debug page images")
    args = ap.parse_args(argv)

    pdf_dir = args.pdf_dir
    out_path = args.out

    if not os.path.isdir(pdf_dir):
        print(f"[ERROR] Folder nie istnieje: {pdf_dir}")
        sys.exit(1)

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[WARN] Brak plików PDF w: {pdf_dir}")

    # Create writer
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for pdf in sorted(pdf_files):
            print(f"[INFO] Przetwarzanie: {os.path.basename(pdf)}")
            try:
                df = process_pdf(pdf, dpi=args.dpi, debug_dir=args.debug_dir)
                sheet = to_excel_sheet_name(pdf)
                df.to_excel(writer, sheet_name=sheet, index=False)
            except Exception as e:
                print(f"[ERROR] Błąd w pliku {pdf}: {e}")
    print(f"[OK] Zapisano: {out_path}")


if __name__ == "__main__":
    main()
