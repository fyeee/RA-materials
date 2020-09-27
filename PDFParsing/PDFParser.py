import sys
import os
import csv
import io
import re
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
from nltk.corpus import stopwords
import nltk
import string


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


def clean_txt_file(txt_path, output_path=None):
    p = nltk.PorterStemmer()
    if not output_path:
        output_path = txt_path
    with open(txt_path, 'rb') as file:
        total_lines = sum(1 for line in file)
    output_fp = open(output_path, 'w')
    with open(txt_path, 'rb') as file:
        count = 0
        line = file.readline().decode('utf-8')
        stop_writing = False
        while line:
            line = re.sub(r"[^\x00-\x7F]", '', line)  # Remove none english characters
            line = line.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            line = line.lower()  # All words to lower case
            line = re.sub(r"[^\s]*[0-9@][^\s]*", '', line)  # Remove all numeric related words
            all_words = line.split(" ")
            write_line = False
            stemmed_words = []
            # Stemming and removing tables
            # Assumption: None table line should contain at least one stop word
            # Assumption: Disclosures only exist in the second half of the reports, and none of the information after
            # that matters
            for i, word in enumerate(all_words):
                stemmed_word = p.stem(word)
                if stemmed_word == "disclosur" and count > (total_lines / 2):
                    stop_writing = True
                if len(word) > 1 and stemmed_word in stopwords.words('english'):
                    write_line = True
                else:
                    stemmed_words.append(stemmed_word)
            if stop_writing:
                break
            if write_line:
                output_fp.write(" ".join(stemmed_words))
            line = file.readline().decode('utf-8')
            count += 1
    output_fp.close()
    with open(output_path, 'rb') as file:
        content = file.read().decode('utf-8')
    with open(output_path, 'w') as file:
        cleaned = re.sub(r"(^| ).(( ).)*( |$)", ' ', content, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*\n+', r'\n', cleaned)
        file.write(cleaned)
    output_fp.close()


output_txt_dir = r".\output_txt\raw_txt"
output_clean_txt_dir = r".\output_txt\clean_txt"
output_csv_dir = r".\output_csv"
input_pdf_dir = r".\input_pdf"

if __name__ == "__main__":
    file_names = [f for f in listdir(input_pdf_dir) if isfile(join(input_pdf_dir, f))]
    for file_name in file_names:
        input_pdf_path = join(input_pdf_dir, file_name)
        output_txt_path = join(output_txt_dir, os.path.splitext(file_name)[0]) + ".txt"
        output_clean_txt_path = join(output_clean_txt_dir, os.path.splitext(file_name)[0]) + ".txt"

        # export_as_txt(input_pdf_path, output_txt_path)
        # export_as_csv(output_txt_path, output_csv_path)
        clean_txt_file(output_txt_path, output_clean_txt_path)