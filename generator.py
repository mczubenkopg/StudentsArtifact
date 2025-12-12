"""
Generator PDF ankiety studenckiej z kratkami do zakreślenia
Wymagania: pip install reportlab qrcode pillow
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import qrcode
from io import BytesIO


class SurveyPDFGenerator:
    def __init__(self, output_filename="ankieta_studencka.pdf"):
        self.output_filename = output_filename
        self.width, self.height = A4
        self.margin = 10 * mm
        self.usable_width = self.width - 2 * self.margin

    def generate_qr_code(self, data):
        """Generuje kod QR i zwraca jako obiekt ImageReader"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=1,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Konwersja do BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        return ImageReader(buffer)

    def draw_header(self, c, y):
        """Rysuje nagłówek ankiety"""
        c.setFont("Helvetica-Bold", 11)
        title = "ANKIETA DLA STUDENTÓW OCENIAJĄCYCH PRZEDMIOTY I NAUCZYCIELI AKADEMICKICH"
        c.drawCentredString(self.width / 2, y, title)
        y -= 5 * mm

        c.setFont("Helvetica", 8)
        scale_text = "Skala: 1-zdecydowanie nie, 2-raczej nie, 3-trudno powiedzieć, 4-raczej tak, 5-zdecydowanie tak, 6-nie wiem"
        c.drawCentredString(self.width / 2, y, scale_text)
        y -= 6 * mm

        return y

    def draw_checkbox_row(self, c, x, y, question_text, question_num):
        """Rysuje wiersz z pytaniem i kratkami 1-6"""
        # Rysowanie pytania
        c.setFont("Helvetica", 7)

        # Podział tekstu na linie jeśli jest za długi
        max_width = self.usable_width - 35 * mm
        words = question_text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            if c.stringWidth(test_line, "Helvetica", 7) < max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Rysowanie tekstu pytania
        question_x = x + 5 * mm
        for i, line in enumerate(lines):
            c.drawString(question_x, y - i * 2.5 * mm, line)

        text_height = len(lines) * 2.5 * mm

        # Rysowanie kratek
        box_size = 3 * mm
        box_spacing = 1 * mm
        start_x = self.width - self.margin - 30 * mm
        box_y = y - 1.5 * mm

        for i in range(1, 7):
            box_x = start_x + (i - 1) * (box_size + box_spacing)

            # Rysowanie kratki
            c.rect(box_x, box_y - box_size, box_size, box_size)

            # Rysowanie numeru nad kratką
            c.setFont("Helvetica", 6)
            c.drawCentredString(box_x + box_size / 2, box_y + 0.5 * mm, str(i))
            c.setFont("Helvetica", 7)

        return max(text_height, 4 * mm)

    def draw_closed_questions(self, c, y):
        """Rysuje część A - pytania zamknięte"""
        c.setFont("Helvetica-Bold", 9)
        c.drawString(self.margin, y, "CZĘŚĆ A: PYTANIA ZAMKNIĘTE")
        y -= 4 * mm

        questions = [
            {
                'section': 'Jakość prowadzenia zajęć',
                'items': [
                    '1.1. Zajęcia były prowadzone w sposób zrozumiały i uporządkowany.',
                    '1.2. Tempo zajęć było odpowiednie dla poziomu grupy.',
                    '1.3. Prowadzący zachęcał do zadawania pytań i aktywnego udziału.',
                    '1.4. Prowadzący był przygotowany merytorycznie do prowadzenia zajęć.'
                ]
            },
            {
                'section': 'Organizacja i materiały dydaktyczne',
                'items': [
                    '2.1. Materiały udostępniane do zajęć były przydatne i aktualne.',
                    '2.2. Zajęcia były zgodne z kartą przedmiotu i zapowiedzianymi treściami.',
                    '2.3. Forma zajęć sprzyjała przyswajaniu wiedzy.',
                    '2.4. Wymagania dotyczące zaliczenia były jasno określone.'
                ]
            },
            {
                'section': 'Kompetencje dydaktyczne i sposób komunikacji',
                'items': [
                    '3.1. Prowadzący przekazywał informacje w sposób przystępny.',
                    '3.2. Prowadzący był otwarty na opinie i sugestie studentów.',
                    '3.3. Atmosfera na zajęciach sprzyjała uczeniu się.',
                    '3.4. Komunikacja między prowadzącym a studentami była jasna i kulturalna.'
                ]
            },
            {
                'section': 'Ocena treści i efektów kształcenia',
                'items': [
                    '4.1. Treści realizowane na zajęciach były interesujące.',
                    '4.2. Treści zajęć były powiązane z praktyką inżynierską.',
                    '4.3. Stopień zaawansowania grupy był adekwatny do prowadzonych zajęć.',
                    '4.4. Poziom trudności zajęć był adekwatny do mojego przygotowania.',
                    '4.5. Zajęcia przyczyniły się do zwiększenia moich kompetencji.',
                    '4.6. Oceniam ten przedmiot jako wartościowy dla mojego kierunku studiów.'
                ]
            }
        ]

        question_num = 0

        for section_data in questions:
            if y < 180 * mm:
                break

            # Rysowanie tytułu sekcji
            c.setFont("Helvetica-Bold", 7)
            c.drawString(self.margin, y, section_data['section'])
            y -= 3 * mm

            c.setFont("Helvetica", 7)

            # Rysowanie pytań w sekcji
            for item in section_data['items']:
                if y < 180 * mm:
                    break

                height = self.draw_checkbox_row(c, self.margin, y, item, question_num)
                y -= height
                question_num += 1

            y -= 2 * mm

        return y

    def draw_open_questions(self, c, y):
        """Rysuje część B - pytania otwarte z QR kodami"""
        y -= 3 * mm

        c.setFont("Helvetica-Bold", 9)
        c.drawString(self.margin, y, "CZĘŚĆ B: PYTANIA OTWARTE")
        y -= 4 * mm

        open_questions = [
            'B1. Co było najmocniejszą stroną tych zajęć?',
            'B2. Co było najsłabszą stroną tych zajęć?',
            'B3. Co należałoby poprawić w sposobie prowadzenia zajęć lub organizacji przedmiotu?',
            'B4. Jakie elementy zajęć były dla Ciebie najbardziej przydatne lub inspirujące?',
            'B5. Jakie elementy merytoryczne należałoby dodać do przedmiotu?'
        ]

        c.setFont("Helvetica", 7)

        box_height = 15 * mm
        qr_size = 12 * mm

        for i, question in enumerate(open_questions):
            if y - box_height < self.margin:
                break

            q_id = f'B{i + 1}'

            # Rysowanie pytania
            c.drawString(self.margin + 2 * mm, y - 3 * mm, question)

            # Rysowanie ramki na odpowiedź
            c.rect(self.margin, y - box_height,
                   self.usable_width - qr_size - 2 * mm,
                   box_height - 4 * mm)

            # Generowanie i rysowanie kodu QR
            qr_img = self.generate_qr_code(q_id)
            c.drawImage(qr_img,
                        self.width - self.margin - qr_size,
                        y - qr_size,
                        width=qr_size,
                        height=qr_size)

            y -= box_height

        return y

    def generate(self):
        """Generuje kompletny PDF ankiety"""
        c = canvas.Canvas(self.output_filename, pagesize=A4)

        y = self.height - self.margin

        # Nagłówek
        y = self.draw_header(c, y)

        # Część A - pytania zamknięte
        y = self.draw_closed_questions(c, y)

        # Część B - pytania otwarte
        y = self.draw_open_questions(c, y)

        # Dodanie informacji w stopce (opcjonalnie)
        c.setFont("Helvetica", 6)
        footer_text = "Ankieta anonimowa - dziękujemy za wypełnienie"
        c.drawCentredString(self.width / 2, 10 * mm, footer_text)

        # Zapisanie PDF
        c.save()
        print(f"✓ Wygenerowano plik: {self.output_filename}")


def main():
    """Główna funkcja programu"""
    print("=" * 60)
    print("GENERATOR PDF ANKIETY STUDENCKIEJ")
    print("=" * 60)

    # Nazwa pliku wyjściowego
    output_file = "ankieta_studencka.pdf"

    # Tworzenie generatora i generowanie PDF
    generator = SurveyPDFGenerator(output_file)
    generator.generate()

    print("\n" + "=" * 60)
    print("PDF WYGENEROWANY POMYŚLNIE")
    print("=" * 60)
    print(f"\nPlik zapisano jako: {output_file}")
    print("\nMożesz teraz:")
    print("1. Wydrukować ankietę")
    print("2. Rozdać studentom do wypełnienia")
    print("3. Zeskanować wypełnione ankiety")
    print("4. Użyć skryptu analizatora do automatycznej obróbki")


if __name__ == "__main__":
    main()