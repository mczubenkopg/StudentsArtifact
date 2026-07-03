from copy import deepcopy

import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image

from tools.final_excel import build_excel
from tools.polish_ocr2 import recognize_handwritten_polish
from tools.checkbox_analyzer import analyse_checkboxes, draw_checkbox_debug
from tools.image_tools import rectify_image
from tools.qrcode_tools import decode_qrcodes
from tools.qwen_recognition import recognize_handwriting


class SingleResult:
    part_a_texts = {
        'A.1.1': 'Zajęcia były prowadzone w sposób zrozumiały i uporządkowany.',
        'A.1.2': 'Tempo zajęć nie było odpowiednie dla poziomu grupy.',
        'A.1.3': 'Prowadzący zachęcał do zadawania pytań i aktywnego udziału.',
        'A.1.4': 'Prowadzący nie był przygotowany merytorycznie do prowadzenia zajęć.',
        'A.1.5': 'Prowadzący dostosowywał treści do uczestników zajęć.',
        'A.2.1': 'Materiały udostępniane do zajęć były przydatne.',
        'A.2.2': 'Materiały udostępniane do zajęć były przestarzałe.',
        'A.2.3': 'Zajęcia nie były zgodne z kartą przedmiotu i zapowiedzianymi treściami.',
        'A.2.4': 'Forma zajęć (W/Ć/L/P/S) sprzyjała przyswajaniu wiedzy.',
        'A.2.5': 'Wymagania dotyczące zaliczenia nie były jasno określone.',
        'A.3.1': 'Prowadzący przekazywał informacje w sposób przystępny.',
        'A.3.2': 'Prowadzący był zamknięty na opinie i sugestie studentów',
        'A.3.3': 'Atmosfera na zajęciach sprzyjała uczeniu się.',
        'A.3.4': 'Komunikacja między prowadzącym a studentami była jasna i kulturalna.',
        'A.4.1': 'Treści realizowane na zajęciach były interesujące.',
        'A.4.2': 'Treści zajęć były powiązane z praktyką inżynierską.',
        'A.4.3': 'Stopień zaawansowania grupy był adekwatny do prowadzonych zajęć.',
        'A.4.4': 'Poziom trudności zajęć był adekwatny do mojego przygotowania.',
        'A.4.5': 'Zajęcia nie przyczyniły się do zwiększenia moich kompetencji.',
        'A.4.6': 'Oceniam ten przedmiot jako wartościowy dla mojego kierunku studiów.',
    }
    scale_desc = [
        '1 – zdecyd. nie',
        '2 – raczej nie',
        '3 – trudno pow.',
        '4 – raczej tak',
        '5 – zdecyd. tak',
        '6 – bez odp.',
    ]
    part_b_texts = [
        'B.1. Co było najmocniejszą stroną tych zajęć?',
        'B.2. Co było najsłabszą stroną tych zajęć?',
        'B.3. Co należałoby poprawić w sposobie prowadzenia zajęć lub organizacji przedmiotu?',
        'B.4. Jakie elementy zajęć były dla Ciebie najbardziej przydatne lub inspirujące?',
        'B.5. Jakie elementy merytoryczne należałoby dodać do przedmiotu?',
    ]
    part_a_column_names = [
        'A.1.1', 'A.1.2', 'A.1.3', 'A.1.4', 'A.1.5',
        'A.2.1', 'A.2.2', 'A.2.3', 'A.2.4', 'A.2.5',
        'A.3.1', 'A.3.2', 'A.3.3', 'A.3.4',
        'A.4.1', 'A.4.2', 'A.4.3', 'A.4.4', 'A.4.5', 'A.4.6'
    ]
    part_b_column_names = ['B.1', 'B.2', 'B.3', 'B.4', 'B.5']


"""
Act like professional python developer. Write a code as simple as possible.
Create an excel file based on the input lists of two elements: a_codes, and b_codes. 
a_codes is a dict with keys 11-15, 21-25, 31-34, 41-46, and values from 1-6. The corresponds to questions and answers from lists:
part_a_texts = {
        'A.1.1': 'Zajęcia były prowadzone w sposób zrozumiały i uporządkowany.',
        'A.1.2': 'Tempo zajęć nie było odpowiednie dla poziomu grupy.',
        'A.1.3': 'Prowadzący zachęcał do zadawania pytań i aktywnego udziału.',
        'A.1.4': 'Prowadzący nie był przygotowany merytorycznie do prowadzenia zajęć.',
        'A.1.5': 'Prowadzący dostosowywał treści do uczestników zajęć.',
        'A.2.1': 'Materiały udostępniane do zajęć były przydatne.',
        'A.2.2': 'Materiały udostępniane do zajęć były przestarzałe.',
        'A.2.3': 'Zajęcia nie były zgodne z kartą przedmiotu i zapowiedzianymi treściami.',
        'A.2.4': 'Forma zajęć (W/Ć/L/P/S) sprzyjała przyswajaniu wiedzy.',
        'A.2.5': 'Wymagania dotyczące zaliczenia nie były jasno określone.',
        'A.3.1': 'Prowadzący przekazywał informacje w sposób przystępny.',
        'A.3.2': 'Prowadzący był zamknięty na opinie i sugestie studentów',
        'A.3.3': 'Atmosfera na zajęciach sprzyjała uczeniu się.',
        'A.3.4': 'Komunikacja między prowadzącym a studentami była jasna i kulturalna.',
        'A.4.1': 'Treści realizowane na zajęciach były interesujące.',
        'A.4.2': 'Treści zajęć były powiązane z praktyką inżynierską.',
        'A.4.3': 'Stopień zaawansowania grupy był adekwatny do prowadzonych zajęć.',
        'A.4.4': 'Poziom trudności zajęć był adekwatny do mojego przygotowania.',
        'A.4.5': 'Zajęcia nie przyczyniły się do zwiększenia moich kompetencji.',
        'A.4.6': 'Oceniam ten przedmiot jako wartościowy dla mojego kierunku studiów.',
    }
    scale_desc = [
        '1 – zdecyd. nie',
        '2 – raczej nie',
        '3 – trudno pow.',
        '4 – raczej tak',
        '5 – zdecyd. tak',
        '6 – bez odp.',
    ]
Get the part_a_texts as the header row (text should be rotated by 90deg), the second row (marked in yellow) should show the average of next rows.  
and fill next rows by each element of list with a value from 1 to 5. 6 should be treated as a lack of information - blank - empty string.
Effectively there should be 20 columns from part a.   
Part b should have 5 columns of text extracted from dict b_codes. Keys are 21-25, while values are tuples of string and np.array (image). Similarly, the columns should have headers: 
part_b_texts = [
        'B.1. Co było najmocniejszą stroną tych zajęć?',
        'B.2. Co było najsłabszą stroną tych zajęć?',
        'B.3. Co należałoby poprawić w sposobie prowadzenia zajęć lub organizacji przedmiotu?',
        'B.4. Jakie elementy zajęć były dla Ciebie najbardziej przydatne lub inspirujące?',
        'B.5. Jakie elementy merytoryczne należałoby dodać do przedmiotu?',
    ]
. The strings should be placed in excel cells, while np.arrays should be converted to image form and pasted into excel in next 5 colums.   
"""


def __init__(self):
    # Definicja kolumn zgodnie z kodem generatora
    self.part_a_answers = []
    self.part_b_answers_text = []
    self.part_b_answers_images = []


def set_part_a(self, part_a_answers):
    self.part_a_answers = part_a_answers


def set_part_b(self, part_b_answers):
    pass


INPUT_FOLDER = Path("./ankiety_2025_26_zima_sem_1")
OUTPUT_FOLDER = Path("./wyniki")


def analyze_surveys():
    """Analizuje wszystkie pliki PDF w folderze"""

    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
    print(f"Znaleziono {len(pdf_files)} plików PDF do analizy...")
    for pdf_file in pdf_files:
        print(f"\nPrzetwarzanie: {pdf_file.name}")
        pdf_list = []
        try:
            # PDF to images with 300 dpi
            images = convert_from_path(pdf_file, dpi=300)
            print(f"  Liczba stron: {len(images)}")
            for page_num, img in enumerate(images[1:]):
                print(f"  Analiza strony {page_num}...")
                try:
                    page_img = rectify_image(np.array(img), dark_threshold=150, search_radius_frac=0.25)
                    pdf_list.append(analyse_content(page_img))
                except Exception as e:
                    print(f"    ✗ Błąd strony {page_num}: {str(e)}")
        except Exception as e:
            print(f"✗ Błąd przy przetwarzaniu {pdf_file.name}: {str(e)}")
        print(f"Running {pdf_file.name} to Excel file")
        build_excel(respondents=pdf_list, output_path=str(OUTPUT_FOLDER.joinpath(pdf_file.name+'.xlsx')))
    return True


def analyse_content(page_img: np.ndarray, debug: bool = False):
    """Qr code finder"""
    drawing_copy = deepcopy(page_img)
    one_three = int(0.35 * page_img.shape[1])
    cv_left_img = cv2.cvtColor(page_img[:, :one_three, :], cv2.COLOR_BGR2RGB)
    cv_right_img = page_img[:, one_three:, :]

    final_a_codes = {}
    a_codes = decode_qrcodes(image=cv_left_img, expected_count=20)
    avg_px = 38
    for code in a_codes:
        avg_px += ((code.bbox[2] - code.bbox[0]) + (code.bbox[3] - code.bbox[1])) / 2
    avg_px /= len(a_codes) + 1
    for code in a_codes:
        analysis = analyse_checkboxes(cv_left_img, code.bbox, mark_threshold=0.15,
                                      gap_ratio=3.3 / 9, avg_qr_size_px=avg_px)
        if debug:
            drawing_copy = draw_checkbox_debug(drawing_copy, analysis)
        final_a_codes[int(code.payload)] = analysis.marked_index
    if debug:
        print(f'Final a codes with results: {final_a_codes}')
    final_b_codes = {}
    drawing_copy_right = deepcopy(cv_right_img)
    b_codes = sorted(decode_qrcodes(image=cv_right_img, expected_count=5), key=lambda x: x.payload)
    avg_px = 0
    for code in b_codes:
        avg_px += ((code.bbox[2] - code.bbox[0]) + (code.bbox[3] - code.bbox[1])) / 2
    avg_px /= len(b_codes)
    h, w, _ = cv_right_img.shape
    for i in range(5):
        if i != 4:
            code_0, code_1 = b_codes[i:i + 2]
            frame_bbox = code_0.bbox[0] - int(avg_px / 4), code_0.bbox[1] - int(avg_px / 4), w - int(avg_px / 2), \
                         code_1.bbox[1] - int(0.75 * avg_px)
        else:
            code_0 = b_codes[i]
            frame_bbox = code_0.bbox[0] - int(avg_px / 4), code_0.bbox[1] - int(avg_px / 4), w - int(avg_px / 2), \
                         h - int(0.75 * avg_px)

        text_box = cv_right_img[frame_bbox[1]:frame_bbox[3], frame_bbox[0]:frame_bbox[2], :]
        try:
            print("Trying qwen")
            r = recognize_handwriting(text_box)
        except Exception as e:
            print("Qwen not work, using tesseract")
            r = recognize_handwritten_polish(text_box)
        key = 'b'+code_0.payload[-1]
        final_b_codes[key] = (r, text_box)
        if debug:
            image = Image.fromarray(drawing_copy)
            image.show('Image')

    final_a_codes.update(final_b_codes)
    return final_a_codes



if __name__ == '__main__':
    analyze_surveys()
