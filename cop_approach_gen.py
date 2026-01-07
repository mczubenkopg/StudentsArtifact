
# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

PAGE_W, PAGE_H = A4

# Layout (margins, gaps)
MARGIN_L = 10*mm
MARGIN_R = 10*mm
MARGIN_T = 10*mm
MARGIN_B = 10*mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

FIDUCIAL_SIZE = 5*mm
CHECKBOX_SIZE = 5*mm
CHECKBOX_GAP = 2*mm
COL_GAP = 6*mm  # trzy kolumny => ciaśniejsza szczelina

# Font z polskimi znakami
FONT_NAME = 'DejaVuSans'
for fp in [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, fp)); break
        except Exception: pass
else:
    FONT_NAME = 'Helvetica'

# Tylko kody (A.1.1 ... A.4.6)
partA_codes = [
    ('Jakość prowadzenia zajęć', ['A.1','A.2','A.3','A.4','A.5']),
    ('Organizacja i materiały dydaktyczne', ['A.6','A.7','A.8','A.9','A.10']),
    ('Kompetencje dydaktyczne i sposób komunikacji', ['A.11','A.12','A.13','A.14']),
    ('Ocena treści i efektów kształcenia', ['A.15','A.16','A.17','A.18','A.19','A.20']),
]
flat_codes = [code for _, arr in partA_codes for code in arr]
col_splits = [flat_codes[:7], flat_codes[7:14], flat_codes[14:]]

# Skrócone, pionowe opisy skali
scale_desc = [
    '1 – zdecyd. nie',
    '2 – raczej nie',
    '3 – trudno pow.',
    '4 – raczej tak',
    '5 – zdecyd. tak',
    '6 – bez odp.',
]

# Część B — etykiety
partB_texts = [
    'B.1. Co było najmocniejszą stroną tych zajęć?',
    'B.2. Co było najsłabszą stroną tych zajęć?',
    'B.3. Co należałoby poprawić w sposobie prowadzenia zajęć lub organizacji przedmiotu?',
    'B.4. Jakie elementy zajęć były dla Ciebie najbardziej przydatne lub inspirujące?',
    'B.5. Jakie elementy merytoryczne należałoby dodać do przedmiotu?',
]

def draw_fiducials(c):
    c.setFillColor(colors.black)
    c.rect(MARGIN_L, PAGE_H - MARGIN_T - FIDUCIAL_SIZE, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)
    c.rect(PAGE_W - MARGIN_R - FIDUCIAL_SIZE, PAGE_H - MARGIN_T - FIDUCIAL_SIZE, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)
    c.rect(MARGIN_L, MARGIN_B, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)
    c.rect(PAGE_W - MARGIN_R - FIDUCIAL_SIZE, MARGIN_B, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)

def draw_title(c):
    c.setFont(FONT_NAME, 12)
    c.drawCentredString(PAGE_W/2, PAGE_H - MARGIN_T - 4*mm, 'Ankieta studencka')
    c.setFont(FONT_NAME, 9)
    c.drawCentredString(PAGE_W/2, PAGE_H - MARGIN_T - 10*mm, 'Część A: Pytania zamknięte (zaznacz 1–6)  |  Część B: Pytania otwarte')

def draw_qr(c, payload, x, y, size_mm=12):
    d = Drawing(size_mm*mm, size_mm*mm); d.add(qr.QrCodeWidget(payload))
    renderPDF.draw(d, c, x, y)

def draw_scale_header_vertical(c, x0, y_top, col_width):
    boxes_total_w = 6*CHECKBOX_SIZE + 5*CHECKBOX_GAP
    bx = x0 + col_width - boxes_total_w
    c.setFont(FONT_NAME, 6.0)
    for i, text in enumerate(scale_desc):
        cx = bx + i*(CHECKBOX_SIZE + CHECKBOX_GAP) + CHECKBOX_SIZE/2
        c.saveState(); c.translate(cx, y_top + 4*mm); c.rotate(90)
        c.drawCentredString(0, 0, text); c.restoreState()

def draw_partA(c, start_y):
    # c.setFont(FONT_NAME, 11); c.drawString(MARGIN_L, start_y, 'A. Pytania zamknięte – 20 pozycji')
    y = start_y - 7*mm
    col_width = (CONTENT_W - 2*COL_GAP) / 3.0 -5*mm
    x_fix = 5*mm
    x_cols = [MARGIN_L+x_fix, MARGIN_L + col_width + COL_GAP+x_fix, MARGIN_L + 2*(col_width + COL_GAP)+x_fix]
    def draw_column(items, x0, y0):
        y = y0
        draw_scale_header_vertical(c, x0-x_fix, y, col_width)
        y -= 8*mm
        c.setFont(FONT_NAME, 7.0)
        for code in items:
            c.drawString(x0, y, code)
            boxes_total_w = 6*CHECKBOX_SIZE + 5*CHECKBOX_GAP
            bx = x0 + col_width - boxes_total_w - x_fix
            by = y - 1*mm
            c.setStrokeColor(colors.black); c.setLineWidth(0.7)
            for i in range(6):
                c.rect(bx + i*(CHECKBOX_SIZE + CHECKBOX_GAP), by - CHECKBOX_SIZE, CHECKBOX_SIZE, CHECKBOX_SIZE, stroke=1, fill=0)
                c.setFont(FONT_NAME, 7)
                c.drawCentredString(bx + i*(CHECKBOX_SIZE + CHECKBOX_GAP) + CHECKBOX_SIZE/2, by + 1.6*mm, str(i+1))
            y = by - CHECKBOX_SIZE - 1.5*mm
            c.setLineWidth(0.25); c.setStrokeColor(colors.lightgrey); c.line(x0, y, x0 + col_width, y)
            y -= 4*mm
        return y
    return min(draw_column(col_splits[0], x_cols[0], y),
               draw_column(col_splits[1], x_cols[1], y),
               draw_column(col_splits[2], x_cols[2], y))

def draw_partB(c, start_y):
    # c.setFont(FONT_NAME, 11); c.drawString(MARGIN_L, start_y, 'B. Pytania otwarte – 5 pozycji')
    y = start_y + 4*mm; box_h = 26*mm; qr_size = 12*mm
    for i, q in enumerate(partB_texts, start=1):
        c.setFont(FONT_NAME, 8.0); c.drawString(MARGIN_L, y, q)
        y_box_top = y - 2.5*mm; c.setLineWidth(1); c.setStrokeColor(colors.black)
        c.rect(MARGIN_L, y_box_top - box_h, CONTENT_W, box_h, stroke=1, fill=0)
        qr_x = MARGIN_L - 3*mm; qr_y = y_box_top - 5*mm - 2*qr_size
        draw_qr(c, f'B.{i}', qr_x, qr_y, size_mm=12)
        left_pad = qr_x + 2*qr_size + 6*mm; c.setStrokeColor(colors.lightgrey); c.setLineWidth(0.2)
        gy = y_box_top - box_h + 1*mm
        while gy < y_box_top - 1.5*mm:
            c.line(left_pad, gy, MARGIN_L + CONTENT_W - 3*mm, gy); gy += 6*mm
        y = y_box_top - box_h - 5*mm
    return y

def build_pdf(filename='ankieta_A4.pdf'):
    c = canvas.Canvas(filename, pagesize=A4)
    draw_fiducials(c); draw_title(c)
    y_after_A = draw_partA(c, PAGE_H - MARGIN_T - 20*mm)
    draw_partB(c, y_after_A - 6*mm)
    c.setFont(FONT_NAME, 7.5); c.setFillColor(colors.black)
    # c.drawString(MARGIN_L, MARGIN_B - 4*mm + FIDUCIAL_SIZE, 'Uwaga: Prosimy pisać drukowanymi literami w polach B i wyraźnie zaznaczać kratki w części A. Skan w 300 DPI.')
    # draw_qr(c, 'SURVEY-PL-2026-A4', PAGE_W - MARGIN_R - FIDUCIAL_SIZE - 18*mm, MARGIN_B + 2*mm, size_mm=12)
    c.showPage(); c.save(); return filename

if __name__ == '__main__':
    print('Wygenerowano plik:', build_pdf())
