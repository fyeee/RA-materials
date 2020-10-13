from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from fuzzywuzzy import fuzz
import pandas as pd
import time
import os


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk


def identify_proper_noun(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    nouns = []
    for item in chunked:
        if type(item) != Tree and item[1] in ['NNP', "NNPS"]:
            nouns.append(item[0])
    return nouns


def identify_company_name(company_names_list, all_possible_name_list, target_company):
    name_list = []
    for possible_name in all_possible_name_list:
        if possible_name not in name_list and fuzz.token_set_ratio(target_company, possible_name) > 90:
            name_list.append(possible_name)
            continue
        for company in company_names_list:
            if possible_name not in name_list and fuzz.ratio(possible_name.lower(), company.lower()) > 85:
                # target_company = company
                name_list.append(possible_name)
    return name_list


def name_removal(line, list_of_names):
    list_of_names.sort(key=len, reverse=True)
    for name in list_of_names:
        line = line.replace(name,'')
    return line


def remove_company_name(all_lines, company_ticker):
    company_df = pd.read_excel("./list_SP500.xlsx", sheet_name="S&P500")
    company_df["S&P 500 Index"] = company_df["S&P 500 Index"].str.slice(0, -4)
    company_names = company_df["S&P 500 Index"]
    target_company = company_df[company_df[".SP500"] == company_ticker]["S&P 500 Index"].max()
    cleaned_lines = []
    for line in all_lines:
        all_possible_names = get_continuous_chunks(line)
        all_possible_names.extend(identify_proper_noun(line))
        recognized_names = identify_company_name(company_names, all_possible_names, target_company)
        if len(recognized_names) > 0:
            line = name_removal(line, recognized_names)
        line = name_removal(line, [company_ticker])
        cleaned_lines.append(line)
    return cleaned_lines


apple_path = r"TestingFiles\2018-01-17-AAPL.OQ-Apple Inc-BofA Global Research-Apple Inc. What’s the playbook for 2018- A bull a...-80603721.txt"
apple2_path = r"TestingFiles\2018-01-26-AAPL.OQ-Apple Inc-BTIG-Apple, Inc. - How Bad Will Apples Guidance Be-80704228.txt"
tsco_path = r"TestingFiles\2018-01-11-TSCO.OQ-Tractor Su-Wedbush Securities I-Downgrading to NEUTRAL; Key Drivers Improving, but...-80541238.txt"
target_path = apple_path