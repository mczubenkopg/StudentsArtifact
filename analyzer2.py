from copy import deepcopy

import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image

from tools.checkbox_analyzer import analyse_checkboxes, draw_checkbox_debug
from tools.image_tools import rectify_image
from tools.qrcode_tools import decode_qrcodes
from tools.textbox_anlyzer import extract_part_b_lines


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

    def __init__(self):
        # Definicja kolumn zgodnie z kodem generatora
        self.part_a_answers = []
        self.part_b_answers_text = []
        self.part_b_answers_images = []

    def set_part_a(self, part_a_answers):
        self.part_a_answers = part_a_answers

    def set_part_b(self, part_b_answers):
        pass


INPUT_FOLDER = Path("./sample")
OUTPUT_EXCEL = "wyniki_ankiet.xlsx"


def analyze_surveys():
    """Analizuje wszystkie pliki PDF w folderze"""

    pdf_files = list(INPUT_FOLDER.glob("*.pdf"))
    results = []
    print(f"Znaleziono {len(pdf_files)} plików PDF do analizy...")
    for pdf_file in pdf_files:
        print(f"\nPrzetwarzanie: {pdf_file.name}")
        try:
            # PDF to images with 300 dpi
            images = convert_from_path(pdf_file, dpi=300)
            print(f"  Liczba stron: {len(images)}")
            for page_num, img in enumerate(images[1:]):
                print(f"  Analiza strony {page_num}...")
                try:
                    result = process_single_page(img)
                    # TODO
                    continue
                    if result:
                        result['plik'] = pdf_file.name
                        result['strona'] = page_num
                        results.append(result)
                        print(f"    ✓ Strona {page_num} przeanalizowana")
                except Exception as e:
                    print(f"    ✗ Błąd strony {page_num}: {str(e)}")

        except Exception as e:
            print(f"✗ Błąd przy przetwarzaniu {pdf_file.name}: {str(e)}")

    if results:
        save_to_excel(results)
        print(f"  Przeanalizowano {len(results)} stron ankiet")
    else:
        print("\n✗ Brak wyników do zapisania")
    return results


def process_single_page(page_img):
    """Przetwarza pojedynczą stronę ankiety"""
    result = SingleResult()
    # Analiza części A - kratki do zakreślenia
    page_img = rectify_image(np.array(page_img), dark_threshold=150, search_radius_frac=0.25)

    part_a, part_b = analyse_content(page_img)

    return result


def analyse_content(page_img: np.ndarray, debug: bool = True):
    """Qr code finder"""
    drawing_copy = deepcopy(page_img)
    one_three = int(0.35 * page_img.shape[1])
    cv_left_img = cv2.cvtColor(page_img[:, :one_three, :], cv2.COLOR_BGR2RGB)
    cv_right_img = cv2.cvtColor(page_img[:, one_three:, :], cv2.COLOR_BGR2RGB)


    a_codes = decode_qrcodes(image=cv_left_img, expected_count=20)
    avg_px = 38
    for code in a_codes:
        avg_px += ((code.bbox[2] - code.bbox[0]) + (code.bbox[3] - code.bbox[1]))/2
    avg_px /= len(a_codes)+1
    final_a_codes = {}
    for code in a_codes:
        analysis = analyse_checkboxes(cv_left_img, code.bbox, mark_threshold=0.15,
                                      gap_ratio=3.3/9, avg_qr_size_px=avg_px)
        if debug:
            drawing_copy = draw_checkbox_debug(drawing_copy, analysis)

        final_a_codes[code.payload] = analysis

    final_b_codes = {}

    b_codes = decode_qrcodes(image=cv_right_img, expected_count=5)
    avg_px = 0
    for code in b_codes:
        avg_px += ((code.bbox[2] - code.bbox[0]) + (code.bbox[3] - code.bbox[1]))/2
    avg_px /= len(b_codes)
    for code in b_codes:
        lines = extract_part_b_lines(cv_right_img, qr_bbox=code.bbox, avg_qr_size_px=avg_px)
        #TODO

    print(lines)
    if debug:
        image = Image.fromarray(drawing_copy)
        image.show('Image')
    return final_a_codes, final_b_codes


"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

image = Image.open("roi.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
"""

def save_to_excel(answers):
    pass


if __name__ == '__main__':
    analyze_surveys()
