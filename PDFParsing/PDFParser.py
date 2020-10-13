import csv
import re
from os import listdir
from os.path import isfile, join
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter, LTChar
from pdfminer.layout import LAParams
from nltk.corpus import stopwords
import nltk
import pandas as pd
from CompanyNameRecognition import *
import time


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


def remove_table_and_disclosure(all_decoded_lines):
    p = nltk.PorterStemmer()
    cleaned_lines = []
    stop_parsing = False
    for i, line in enumerate(all_decoded_lines):
        write_line = False
        all_words = line.split(" ")
        for word in all_words:
            stemmed_word = p.stem(word)
            if stemmed_word == "disclosur" and i > (len(all_decoded_lines) / 2):
                stop_parsing = True
            if len(word) > 1 and word in stopwords.words('english'):
                write_line = True
        if write_line:
            cleaned_lines.append(line)
        if stop_parsing:
            break
    return cleaned_lines


def pre_processing(all_lines):
    """
    Removing stop words,
    Replace all words with lower-case
    Remove punctuation
    Remove all numeric related words
    """
    p = nltk.PorterStemmer()
    cleaned_lines = []
    for line in all_lines:
        line = re.sub(r"[^\s]*[0-9@][^\s]*", '', line)  # Remove all numeric related words
        line = line.translate(dict((ord(char), " ") for char in ",.!:;@#$%^*()+_=~?<>\"'/\\"))  # Remove punctuation
        line = line.lower()  # All words to lower case
        for key in phrases_mapping:
            if key in line:
                line = re.sub(key, phrases_mapping[key][0], line)
        all_words = line.split(" ")
        stemmed_words = []
        # Stemming and removing stop words
        for i, word in enumerate(all_words):
            stemmed_word = p.stem(word)
            if word not in stopwords.words('english'):
                stemmed_words.append(stemmed_word)
        cleaned_lines.append(" ".join(stemmed_words))
    return cleaned_lines


def clean_txt_file(txt_path, output_path=None):
    report_name = os.path.splitext(txt_path)[0]
    company_ticker = report_name.split("-")[3]
    if not output_path:
        output_path = txt_path
    output_fp = open(output_path, 'w')
    with open(txt_path, 'rb') as file:
        all_decoded_lines = []
        line = file.readline().decode('utf-8')
        while line:
            line = re.sub(r"[^\x00-\x7F]", '', line)  # Remove none english characters
            line = line.replace('\r', '')
            all_decoded_lines.append(line)
            line = file.readline().decode('utf-8')
        all_decoded_lines = remove_table_and_disclosure(all_decoded_lines)
        no_company_lines = remove_company_name(all_decoded_lines, company_ticker)
        all_clean_lines = pre_processing(no_company_lines)
        for line in all_clean_lines:
            output_fp.write(line)
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
phrases_mapping = pd.read_csv(r'.\dictionaries\list_of_high-frequency_phrases.txt', sep="	", header=None).set_index(0).T.to_dict('list')


if __name__ == "__main__":
    start = time.time()
    file_names = [f for f in listdir(input_pdf_dir) if isfile(join(input_pdf_dir, f))]
    for file_name in file_names:
        input_pdf_path = join(input_pdf_dir, file_name)
        output_txt_path = join(output_txt_dir, os.path.splitext(file_name)[0]) + ".txt"
        output_clean_txt_path = join(output_clean_txt_dir, os.path.splitext(file_name)[0]) + ".txt"

        # export_as_txt(input_pdf_path, output_txt_path)
        # export_as_csv(output_txt_path, output_csv_path)
        clean_txt_file(output_txt_path, output_clean_txt_path)
    end = time.time()
    print(end - start)