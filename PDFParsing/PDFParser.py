import sys
import os
import csv
import io
from os import listdir
from os.path import isfile, join
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter, LTChar
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter


class CsvConverter(TextConverter):
    def __init__(self, *args, **kwargs):
        TextConverter.__init__(self, *args, **kwargs)

    def end_page(self, i):
        from collections import defaultdict
        lines = defaultdict(lambda: {})
        for child in self.cur_item._objs:  # <-- changed
            if isinstance(child, LTChar):
                (_, _, x, y) = child.bbox
                line = lines[int(-y)]
                line[x] = child._text.encode('utf-8')  # <-- changed

        for y in sorted(lines.keys()):
            try:
                line = lines[y]
                self.outfp.write(";".join(line[x] for x in sorted(line.keys())))
                self.outfp.write("\n")
            except:
                pass

def export_as_csv(txt_path, output_csv_path):
    with open(txt_path, 'r', encoding='utf-8') as in_file:
        stripped = [line.strip() for line in in_file][:-1]
        lines = (line.split(",") for line in stripped if line)
        with open(output_csv_path, 'w', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)


def export_as_txt(pdf_path, output_txt_path):
    output = open(output_txt_path, 'w', encoding='utf-8')
    resource_manager = PDFResourceManager()
    page_numbers = set()
    device = TextConverter(resource_manager, output, laparams = LAParams())
    with open(pdf_path, 'rb') as fp:
        interpreter = PDFPageInterpreter(resource_manager, device)
        for page in PDFPage.get_pages(fp, page_numbers):
            interpreter.process_page(page)
    device.close()
    output.close()


output_txt_dir = r".\output_txt"
output_csv_dir = r".\output_csv"
input_pdf_dir = r".\input_pdf"

if __name__ == "__main__":
    file_names = [f for f in listdir(input_pdf_dir) if isfile(join(input_pdf_dir, f))]
    for file_name in file_names:
        input_pdf_path = join(input_pdf_dir, file_name)
        output_txt_path = join(output_txt_dir, os.path.splitext(file_name)[0]) + ".txt"
        output_csv_path = join(output_csv_dir, os.path.splitext(file_name)[0]) + ".csv"

        export_as_txt(input_pdf_path, output_txt_path)
        # export_as_csv(output_txt_path, output_csv_path)
