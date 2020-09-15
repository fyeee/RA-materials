import sys
import os
import csv
import io
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter


def export_as_csv(pdf_path, output_csv_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    counter = 1
    with open(output_csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for page in extract_text_by_page(pdf_path):
            text = page
            words = text.split()
            writer.writerow(words)


def extract_text_by_page(pdf_path):
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
            yield text
            converter.close()
            fake_file_handle.close()


def export_as_txt(pdf_path, output_txt_path):
    output = open(output_txt_path, 'w', encoding='utf-8')
    resourse_manager = PDFResourceManager()
    page_numbers = set()
    device = TextConverter(resourse_manager, output, laparams = LAParams())
    with open(pdf_path, 'rb') as fp:
        interpreter = PDFPageInterpreter(resourse_manager, device)
        for page in PDFPage.get_pages(fp, page_numbers):
            interpreter.process_page(page)
    device.close()
    output.close()

output_txt_path = r"C:\Users\frank\Desktop\research-materials\PDFParsing\txt_sample.txt"
output_csv_path = r"C:\Users\frank\Desktop\research-materials\PDFParsing\csv_sample.csv"
pdf_path = r"C:\Users\frank\Desktop\research-materials\PDFParsing\sample-pdf-file.pdf"

if __name__ == "__main__":
    export_as_txt(pdf_path, output_txt_path)

    export_as_csv(pdf_path, output_csv_path)
