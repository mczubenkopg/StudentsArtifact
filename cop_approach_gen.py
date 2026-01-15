# -*- coding: utf-8 -*-
from pyexpat.model import XML_CTYPE_ANY

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab_qrcode import QRCodeImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

PAGE_W, PAGE_H = A4

# Layout (margins, gaps)
MARGIN_L = 5 * mm
MARGIN_R = 5 * mm
MARGIN_T = 5 * mm
MARGIN_B = 5 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_L - 2 * MARGIN_R

COL_GAP = 6 * mm
TEXT_GAP = 2
CHECKBOX_GAP = 2.5 * mm

FONT_SIZE_LARGE = 12
FONT_SIZE_NORMAL = 8
FONT_SIZE_SMALL = 7

CONTENT_PART_A = CONTENT_W / 3
CONTENT_PART_B = 2 * CONTENT_W / 3 - COL_GAP
TITLE_H = FONT_SIZE_LARGE + TEXT_GAP + 2 * (FONT_SIZE_NORMAL + TEXT_GAP)

X_A = 2 * MARGIN_L
Y_T = PAGE_H - MARGIN_T - TITLE_H
X_B = 2 * MARGIN_L + CONTENT_PART_A + COL_GAP

FIDUCIAL_SIZE = 5 * mm
CHECKBOX_SIZE = 5 * mm

# Font z polskimi znakami
FONT_NAME = 'DejaVuSans'
for fp in [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, fp));
            break
        except Exception:
            pass
else:
    FONT_NAME = 'Helvetica'


def draw_fiducials(c):
    c.setFillColor(colors.black)
    c.rect(MARGIN_L, PAGE_H - MARGIN_T - FIDUCIAL_SIZE, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)
    c.rect(PAGE_W - MARGIN_R - FIDUCIAL_SIZE, PAGE_H - MARGIN_T - FIDUCIAL_SIZE, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0,
           fill=1)
    c.rect(MARGIN_L, MARGIN_B, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)
    c.rect(PAGE_W - MARGIN_R - FIDUCIAL_SIZE, MARGIN_B, FIDUCIAL_SIZE, FIDUCIAL_SIZE, stroke=0, fill=1)


def draw_title(c, questions=False, title='Ankieta studencka'):
    c.setFont(FONT_NAME, FONT_SIZE_LARGE)
    if questions:
        c.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN_T, title)
        c.setFont(FONT_NAME, FONT_SIZE_NORMAL)
        c.drawCentredString(X_A + CONTENT_W / 2, PAGE_H - MARGIN_T - 4 * mm - 12, 'Część A: Pytania zamknięte')
    else:
        c.drawString(X_A + 4, PAGE_H - MARGIN_T, title)
        c.setFont(FONT_NAME, FONT_SIZE_NORMAL)
        c.drawString(X_A+4, PAGE_H - MARGIN_T - 4 * mm - 4, 'Przedmiot i data: '+90*'..')
        c.drawCentredString(X_A + CONTENT_PART_A / 2, PAGE_H - MARGIN_T - 4 * mm - 24, 'Część A: Pytania zamknięte')
        c.line(X_B - COL_GAP / 2, PAGE_H - MARGIN_T - MARGIN_B - TEXT_GAP - 12, X_B - COL_GAP / 2, 2 * MARGIN_B)
        c.drawCentredString(X_B + CONTENT_PART_B / 2, PAGE_H - MARGIN_T - 4 * mm - 24, 'Część B: Pytania otwarte')

    c.setFont(FONT_NAME, FONT_SIZE_SMALL)
    if questions:
        c.drawCentredString(X_A + CONTENT_W / 2, PAGE_H - MARGIN_T - 4 * mm - 24, '(zaznacz 1–6)  ')
    else:
        c.drawCentredString(X_A + CONTENT_PART_A / 2, PAGE_H - MARGIN_T - 4 * mm - 36, '(zaznacz 1–6)  ')
        c.drawCentredString(X_B + CONTENT_PART_B / 2, PAGE_H - MARGIN_T - 4 * mm - 36, '(wypełnij drukowanymi)  ')


def draw_part_a(c, part_a_codes, scale_desc=None, draw_boxes=True):
    c.setFont(FONT_NAME, FONT_SIZE_NORMAL)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    x_0 = X_A

    # Header boxes names
    box_x = x_0 + 12 * mm
    box_dsc_y = Y_T - 28 * mm
    if draw_boxes:
        for i, text in enumerate(scale_desc):
            cx = box_x + i * (CHECKBOX_SIZE + CHECKBOX_GAP) + CHECKBOX_SIZE / 2 + FONT_SIZE_NORMAL / 2 - 2 + CHECKBOX_SIZE
            c.saveState()
            c.translate(cx, box_dsc_y)
            c.rotate(90)
            c.drawString(0, 0, text)
            c.restoreState()
    y = box_dsc_y - 2 * FONT_SIZE_LARGE

    for name, codes in part_a_codes:
        if draw_boxes:
            c.line(X_A, y + CHECKBOX_SIZE - CHECKBOX_GAP, X_A + CONTENT_PART_A + COL_GAP / 2,
                   y + CHECKBOX_SIZE - CHECKBOX_GAP)
        else:
            c.line(X_A, y + CHECKBOX_SIZE - CHECKBOX_GAP, X_A + CONTENT_W,
                   y + CHECKBOX_SIZE - CHECKBOX_GAP)
        y = y - CHECKBOX_GAP
        c.drawString(x_0, y, name)
        y = y - FONT_SIZE_LARGE
        for code in codes:
            bx = box_x+CHECKBOX_SIZE
            by = y
            if draw_boxes:
                # podpisy boxów i boxy
                for i in range(6):
                    c.setFont(FONT_NAME, FONT_SIZE_SMALL)
                    c.rect(bx + i * (CHECKBOX_SIZE + CHECKBOX_GAP), by - CHECKBOX_SIZE, CHECKBOX_SIZE, CHECKBOX_SIZE,
                           stroke=1,
                           fill=0)
                    c.drawCentredString(bx + i * (CHECKBOX_SIZE + CHECKBOX_GAP) + CHECKBOX_SIZE / 2, by + 2, str(i + 1))
            y = y - 1.25 * FONT_SIZE_SMALL

            c.setFont(FONT_NAME, FONT_SIZE_NORMAL)
            if draw_boxes:
                QRCodeImage(code.strip('A').replace('.',''), size=CHECKBOX_SIZE+4*mm).drawOn(c, x_0+2*mm+CHECKBOX_SIZE, y-4*mm)
                c.drawString(x_0, y, code)
            else:
                c.drawString(x_0, y, code)
            y = y - CHECKBOX_SIZE - CHECKBOX_GAP


def draw_part_b(c, part_b_texts):
    x = X_B
    y = Y_T - 24
    box_h = 45 * mm
    qr_size = 20 * mm
    c.setFont(FONT_NAME, FONT_SIZE_NORMAL)
    c.setLineWidth(1)
    c.setStrokeColor(colors.black)
    for i, q in enumerate(part_b_texts, start=1):
        c.drawString(x, y, q)
        y_box_top = y - 2.5 * mm
        c.rect(x, y_box_top - box_h, CONTENT_PART_B, box_h, stroke=1, fill=0)
        qr_x = x + 2
        qr_y = y_box_top - qr_size - 1
        QRCodeImage(f'B.{i}', size=qr_size).drawOn(c, qr_x, qr_y)
        left_pad = qr_x + qr_size + 2
        # lines
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(0.2)
        line_y = y_box_top - 1 * mm - 2 * FONT_SIZE_LARGE
        while abs(y_box_top - line_y) < box_h:
            if abs(line_y - y_box_top) < qr_size + 2:
                c.line(left_pad, line_y, X_B + CONTENT_PART_B - 3 * mm, line_y)
            else:
                c.line(x + 2 * mm, line_y, X_B + CONTENT_PART_B - 3 * mm, line_y)
            line_y -= 2 * FONT_SIZE_LARGE
        c.setLineWidth(1)
        c.setStrokeColor(colors.black)
        y = y_box_top - box_h - 5 * mm
    return y


def draw_foot(c,
              content='Uwaga: Prosimy pisać drukowanymi literami w polach B i wyraźnie zamalować kratki w części A.'):
    c.setFont(FONT_NAME, 7.5)
    c.setFillColor(colors.black)
    c.drawCentredString(PAGE_W / 2, MARGIN_B - 4 * mm + FIDUCIAL_SIZE, content)


def build_student(filename='ankieta_studencka.pdf'):
    part_a_codes_only = [
        ('A.1 Jakość prowadzenia zajęć', ['A.1.1', 'A.1.2', 'A.1.3', 'A.1.4', 'A.1.5']),
        ('A.2 Organizacja zajęć', ['A.2.1', 'A.2.2', 'A.2.3', 'A.2.4', 'A.2.5']),
        ('A.3 Komunikacja i kompetencje', ['A.3.1', 'A.3.2', 'A.3.3', 'A.3.4']),
        ('A.4 Treści i efekty kształcenia', ['A.4.1', 'A.4.2', 'A.4.3', 'A.4.4', 'A.4.5', 'A.4.6']),
    ]
    part_a_codes_questions = [
        ('A.1 Jakość prowadzenia zajęć', ['A.1.1 Zajęcia były prowadzone w sposób zrozumiały i uporządkowany.',
                                          'A.1.2 Tempo zajęć nie było odpowiednie dla poziomu grupy.',
                                          'A.1.3 Prowadzący zachęcał do zadawania pytań i aktywnego udziału.',
                                          'A.1.4 Prowadzący nie był przygotowany merytorycznie do prowadzenia zajęć.',
                                          'A.1.5 Prowadzący dostosowywał treści do uczestników zajęć.']),
        ('A.2 Organizacja zajęć', ['A.2.1 Materiały udostępniane do zajęć były przydatne.',
                                   'A.2.2 Materiały udostępniane do zajęć były przestarzałe.',
                                   'A.2.3 Zajęcia nie były zgodne z kartą przedmiotu i zapowiedzianymi treściami.',
                                   'A.2.4 Forma zajęć (W/Ć/L/P/S) sprzyjała przyswajaniu wiedzy.',
                                   'A.2.5 Wymagania dotyczące zaliczenia nie były jasno określone.']),
        ('A.3 Komunikacja i kompetencje', ['A.3.1 Prowadzący przekazywał informacje w sposób przystępny.',
                                           'A.3.2 Prowadzący był zamknięty na opinie i sugestie studentów',
                                           'A.3.3 Atmosfera na zajęciach sprzyjała uczeniu się.',
                                           'A.3.4 Komunikacja między prowadzącym a studentami była jasna i kulturalna.']),
        ('A.4 Treści i efekty kształcenia', ['A.4.1 Treści realizowane na zajęciach były interesujące.',
                                             'A.4.2 Treści nie zajęć były powiązane z praktyką inżynierską.',
                                             'A.4.3 Stopień zaawansowania grupy był adekwatny do prowadzonych zajęć.',
                                             'A.4.4 Poziom trudności zajęć był adekwatny do mojego przygotowania.',
                                             'A.4.5 Zajęcia nie przyczyniły się do zwiększenia moich kompetencji.',
                                             'A.4.6 Oceniam ten przedmiot jako wartościowy dla mojego kierunku studiów.']),
    ]
    scale_desc = [
        '1 – zdecyd. nie',
        '2 – raczej nie',
        '3 – trudno pow.',
        '4 – raczej tak',
        '5 – zdecyd. tak',
        '6 – bez odp.',
    ]

    # Część B — etykiety
    part_b_texts = [
        'B.1. Co było najmocniejszą stroną tych zajęć?',
        'B.2. Co było najsłabszą stroną tych zajęć?',
        'B.3. Co należałoby poprawić w sposobie prowadzenia zajęć lub organizacji przedmiotu?',
        'B.4. Jakie elementy zajęć były dla Ciebie najbardziej przydatne lub inspirujące?',
        'B.5. Jakie elementy merytoryczne należałoby dodać do przedmiotu?',
    ]
    c = canvas.Canvas(filename, pagesize=A4)
    # Answers
    draw_fiducials(c)
    draw_title(c)
    draw_part_a(c, part_a_codes_only, scale_desc)
    draw_part_b(c, part_b_texts)
    draw_foot(c)
    c.showPage()
    # Questions
    draw_fiducials(c)
    draw_title(c, questions=True)
    draw_part_a(c, part_a_codes_questions, draw_boxes=False)
    draw_foot(c, content='Uwaga: Prosimy nie pisać po kartce z pytaniami.')
    c.showPage()
    c.save()
    return filename


def build_teacher(filename='ankieta_nauczyciela.pdf'):
    part_a_codes_only = [
        ('A.1 Zaangażowanie studentów', ['A.1.1', 'A.1.2', 'A.1.3', 'A.1.4', 'A.1.5']),
        ('A.2 Kompetencje merytoryczne', ['A.2.1', 'A.2.2', 'A.2.3', 'A.2.4', 'A.2.5', 'A.2.6']),
        ('A.3 Kultura osobista', ['A.3.1', 'A.3.2', 'A.3.3', 'A.3.4']),
        ('A.4 Realizacja powierzonych zadań', ['A.4.1', 'A.4.2', 'A.4.3', 'A.4.4', 'A.4.5']),
    ]
    part_a_codes_questions = [
        ('A.1 Zaangażowanie studentów', '''A.1.1 Studenci aktywnie uczestniczyli w zajęciach.
A.1.2 Zadawali pytania i okazywali zainteresowanie tematyką.
A.1.3 Byli przygotowani do zajęć (np. przeczytane materiały, wykonane zadania).
A.1.4 Regularnie uczęszczali na zajęcia.
A.1.5 Wykazywali inicjatywę i chęć podejmowania dodatkowych aktywności.'''.replace(
            '\t', '').split('\n')),
        ('A.2 Kompetencje merytoryczne', '''A.2.1 Studenci posiadali wystarczającą wiedzę wprowadzającą do tematyki zajęć.
A.2.2 Sprawnie przyswajali nowe zagadnienia.
A.2.3 Potrafili samodzielnie rozwiązywać zlecone problemy.
A.2.4 Wykazywali umiejętność krytycznego myślenia.
A.2.5 Umieli połączyć teorię z praktyką inżynierską.
A.2.6 Studenci nadmiernie korzystali z LLM.'''.replace('\t', '').split(
            '\n')),
        ('A.3 Kultura osobista', '''A.3.1 Studenci współpracowali w sposób konstruktywny.
A.3.2 Studenci wykazywali wzajemny szacunek i kulturę osobistą.		
A.3.3 Komunikacja w grupie była przejrzysta i poprawna.
A.3.4 Studenci respektowali zasady obowiązujące na zajęciach.'''.replace('\t',
                                                                                                             '').split(
            '\n')),
        ('A.4 Realizacja powierzonych zadań', '''A.4.1 Terminowo oddawali zadania i projekty.
A.4.2 Jakość wykonania zadań była na zadowalającym poziomie.
A.4.3 Studenci wykazywali kreatywność w rozwiązywaniu problemów.
A.4.4 W pracach projektowych studenci potrafili dzielić się zadaniami.
A.4.5 Całościowa postawa studentów była profesjonalna i odpowiedzialna.'''.replace(
            '\t', '').split('\n')),

    ]

    scale_desc = [
        '1 – nikt',
        '2 – kilkoro',
        '3 – połowa',
        '4 – większość',
        '5 – wszyscy',
        '6 – bez odp.',
    ]

    # Część B — etykiety
    part_b_texts = [
        'B.1. Jak oceniasz najmocniejsze strony tej grupy studenckiej?',
        'B.2. Jakie obszary wymagają poprawy lub dodatkowego wsparcia?',
        'B.3. Jakie działania dydaktyczne mogłyby pomóc w podniesieniu poziomu studentów?',
        'B.4. Jaki jest ogólny poziom widzy grupy?',
        'B.5. Jakie problemy nastąpiły podczas prowadzonych zajęć?'
    ]
    c = canvas.Canvas(filename, pagesize=A4)
    # Answers
    draw_fiducials(c)
    draw_title(c, title='Ankieta nauczyciela')
    draw_part_a(c, part_a_codes_only, scale_desc)
    draw_part_b(c, part_b_texts)
    draw_foot(c)
    c.showPage()
    # Questions
    draw_fiducials(c)
    draw_title(c, title='Ankieta nauczyciela', questions=True)
    draw_part_a(c, part_a_codes_questions, draw_boxes=False)
    draw_foot(c, content='Uwaga: Prosimy nie pisać po kartce z pytaniami.')
    c.showPage()
    c.save()
    return filename


if __name__ == '__main__':
    print('Wygenerowano plik:', build_student())
    print('Wygenerowano plik:', build_teacher())
