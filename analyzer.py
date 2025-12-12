"""
Analizator wypełnionych ankiet studenckich
Wymagania: pip install opencv-python pytesseract pandas Pillow pdf2image pyzbar openpyxl numpy
"""

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from pyzbar.pyzbar import decode
import pandas as pd
from pathlib import Path
from PIL import Image
import os


class SurveyAnalyzer:
    def __init__(self, input_folder, output_excel):
        self.input_folder = Path(input_folder)
        self.output_excel = output_excel
        self.results = []

    def analyze_surveys(self):
        """Analizuje wszystkie pliki PDF w folderze"""
        pdf_files = list(self.input_folder.glob("*.pdf"))

        print(f"Znaleziono {len(pdf_files)} plików PDF do analizy...")

        for pdf_file in pdf_files:
            print(f"\nPrzetwarzanie: {pdf_file.name}")
            try:
                result = self.process_single_survey(pdf_file)
                if result:
                    result['nazwa_pliku'] = pdf_file.name
                    self.results.append(result)
                    print(f"✓ Pomyślnie przeanalizowano {pdf_file.name}")
            except Exception as e:
                print(f"✗ Błąd przy przetwarzaniu {pdf_file.name}: {str(e)}")

        if self.results:
            self.save_to_excel()
            print(f"\n✓ Wyniki zapisano do {self.output_excel}")
        else:
            print("\n✗ Brak wyników do zapisania")

    def process_single_survey(self, pdf_path):
        """Przetwarza pojedynczą ankietę PDF"""
        # Konwersja PDF na obraz
        images = convert_from_path(pdf_path, dpi=300)
        if not images:
            return None

        img = np.array(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        result = {}

        # Analiza części A - kratki do zakreślenia
        closed_answers = self.analyze_closed_questions(gray)
        for q_num, answer in closed_answers.items():
            result[q_num] = answer

        # Analiza części B - pytania otwarte z QR kodami
        open_answers = self.analyze_open_questions(img, gray)
        for q_id, data in open_answers.items():
            result[f"{q_id}_text"] = data['text']
            result[f"{q_id}_image"] = data['image']

        return result

    def analyze_closed_questions(self, gray_img):
        """Analizuje zakreślone kratki w pytaniach zamkniętych (1-6)"""
        height, width = gray_img.shape
        answers = {}

        # Parametry do wykrywania kratek
        # Zakładamy, że kratki są w prawej części strony
        right_margin = int(width * 0.75)

        # Binaryzacja obrazu
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

        # Wykrywanie konturów
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sortowanie konturów od góry do dołu
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filtrowanie małych konturów i tych po lewej stronie
            if 20 < w < 50 and 20 < h < 50 and x > right_margin:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimalna powierzchnia kratki
                    boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: b[1])  # Sortowanie po Y

        # Grupowanie kratek w wiersze (pytania)
        rows = []
        current_row = []
        last_y = -100

        for box in boxes:
            x, y, w, h = box
            if abs(y - last_y) < 30:  # Ta sama linia
                current_row.append(box)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
            last_y = y

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        # Analiza każdego wiersza kratek
        question_sections = [
            ('1.1', '1.2', '1.3', '1.4'),
            ('2.1', '2.2', '2.3', '2.4'),
            ('3.1', '3.2', '3.3', '3.4'),
            ('4.1', '4.2', '4.3', '4.4', '4.5', '4.6')
        ]

        all_questions = []
        for section in question_sections:
            all_questions.extend(section)

        for idx, row in enumerate(rows[:len(all_questions)]):
            if idx >= len(all_questions):
                break

            q_id = all_questions[idx]

            # Sprawdzenie, która kratka jest zakreślona
            max_fill = 0
            selected = 0

            for box_idx, (x, y, w, h) in enumerate(row[:6]):
                # Pobierz region kratki
                roi = binary[y:y + h, x:x + w]
                if roi.size == 0:
                    continue

                # Oblicz procent wypełnienia
                fill_ratio = np.sum(roi > 0) / roi.size

                if fill_ratio > max_fill:
                    max_fill = fill_ratio
                    selected = box_idx + 1  # 1-6

            # Jeśli wypełnienie > 30%, uznajemy za zakreślone
            if max_fill > 0.3:
                answers[q_id] = selected
            else:
                answers[q_id] = None

        return answers

    def analyze_open_questions(self, img, gray_img):
        """Analizuje pytania otwarte z QR kodami"""
        results = {}

        # Dekodowanie QR kodów
        qr_codes = decode(img)

        height, width = gray_img.shape

        # Przygotowanie obrazu do OCR
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Znajdowanie ramek (prostokątów)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filtrowanie - szukamy dużych prostokątów (ramki dla odpowiedzi)
            if w > width * 0.5 and 100 < h < 300:
                rectangles.append((x, y, w, h, y))  # Dodajemy y do sortowania

        rectangles.sort(key=lambda r: r[4])  # Sortowanie od góry do dołu

        # Mapowanie QR kodów do ramek
        qr_map = {}
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            qr_rect = qr.rect
            qr_y = qr_rect.top

            qr_map[qr_data] = qr_y

        # Przetwarzanie każdej ramki
        question_ids = ['B1', 'B2', 'B3', 'B4', 'B5']

        for idx, (x, y, w, h, _) in enumerate(rectangles[:5]):
            if idx >= len(question_ids):
                break

            q_id = question_ids[idx]

            # Wycinanie regionu z odpowiedzią
            roi = gray_img[y:y + h, x:x + w]

            # OCR
            config = '--oem 3 --psm 6 -l pol'
            text = pytesseract.image_to_string(roi, config=config)
            text = text.strip()

            # Zapisanie wyciętego obrazu
            roi_pil = Image.fromarray(roi)

            results[q_id] = {
                'text': text,
                'image': roi_pil
            }

        return results

    def save_to_excel(self):
        """Zapisuje wyniki do pliku Excel"""
        # Przygotowanie danych
        data_rows = []

        for result in self.results:
            row = {'Nazwa pliku': result.get('nazwa_pliku', '')}

            # Pytania zamknięte
            questions = ['1.1', '1.2', '1.3', '1.4',
                         '2.1', '2.2', '2.3', '2.4',
                         '3.1', '3.2', '3.3', '3.4',
                         '4.1', '4.2', '4.3', '4.4', '4.5', '4.6']

            for q in questions:
                row[f'Pyt_{q}'] = result.get(q, '')

            # Pytania otwarte - tekst
            for i in range(1, 6):
                q_id = f'B{i}'
                row[f'Pytanie_otwarte_{q_id}'] = result.get(f'{q_id}_text', '')

            data_rows.append(row)

        # Tworzenie DataFrame
        df = pd.DataFrame(data_rows)

        # Zapisanie do Excel z obrazkami
        with pd.ExcelWriter(self.output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Wyniki', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Wyniki']

            # Dodawanie obrazków
            from openpyxl.drawing.image import Image as XLImage

            img_col_start = len(df.columns) + 2

            for row_idx, result in enumerate(self.results, start=2):
                for i in range(1, 6):
                    q_id = f'B{i}'
                    if f'{q_id}_image' in result:
                        img = result[f'{q_id}_image']

                        # Zapisz tymczasowo
                        temp_path = f'temp_{row_idx}_{q_id}.png'
                        img.save(temp_path)

                        # Dodaj do Excel
                        xl_img = XLImage(temp_path)
                        xl_img.width = 200
                        xl_img.height = 100

                        col_letter = chr(65 + img_col_start + i - 1)
                        cell = f'{col_letter}{row_idx}'
                        worksheet.add_image(xl_img, cell)

                        # Usuń tymczasowy plik
                        os.remove(temp_path)

            # Dostosowanie szerokości kolumn
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width


def main():
    """Główna funkcja programu"""
    # Konfiguracja
    INPUT_FOLDER = "ankiety_wypelnione"  # Folder ze skanami
    OUTPUT_EXCEL = "wyniki_ankiet.xlsx"

    # Tworzenie folderu jeśli nie istnieje
    Path(INPUT_FOLDER).mkdir(exist_ok=True)

    print("=" * 60)
    print("ANALIZATOR ANKIET STUDENCKICH")
    print("=" * 60)

    # Uruchomienie analizy
    analyzer = SurveyAnalyzer(INPUT_FOLDER, OUTPUT_EXCEL)
    analyzer.analyze_surveys()

    print("\n" + "=" * 60)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 60)


if __name__ == "__main__":
    main()