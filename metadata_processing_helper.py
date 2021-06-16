# functions to organize data before building matrix
# -------------------------------------------------
import math
import pandas as pd
import numpy as np
import os

def construct_indu_index_mapping(df):
    """
    Construct a dictionary with
    key: industry code
    value: indexes of all reports in the dataframe
    """
    industries_to_index = {}
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        if math.isnan(row["ggroup"]):
            continue
        industries_to_index[int(row["ggroup"])] = industries_to_index.get(int(row["ggroup"]), set())
        industries_to_index[int(row["ggroup"])].add(i)
    return industries_to_index


def construct_quar_index_mapping(df):
    """
    Construct a dictionary with
    key: quarter
    value: indexes of all reports in the dataframe
    """
    quarter_to_index = {}
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        quarter = row["year"].astype("str") + " q" + row["quarter"].astype("str")
        quarter_to_index[quarter] = quarter_to_index.get(quarter, set())
        quarter_to_index[quarter].add(i)
    return quarter_to_index


def construct_analyst_index_mapping(df, all_files_dcns):
    """
    Construct a dictionary with
    key: analyst
    value: indexes of all reports in the dataframe with the given DCNs(unique identification code for the reports)
    """
    analyst_to_index = {}
    for i, (_, dcn) in enumerate(all_files_dcns):
        analyst = max(df[df["DCN"] == dcn]["Analyst"])
        if not analyst is np.nan:
            analyst_to_index[analyst] = analyst_to_index.get(analyst, []) + [i]
    return analyst_to_index


def get_industry_dcns(df, industries_to_index, industry):
    indexes = industries_to_index[industry]
    # DCN is the unique identification code for the reports

    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns = get_file_paths(dcns)
    return all_files_dcns


def get_industry_quarter_dcns(df, industries_to_index, quarter_to_index, industry, quarter):
    # select all report indices (rows in metadata) for the industry-quarter
    indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
    # set of all company names for the industry-quarter
    all_companies = df.iloc[list(indexes), :]['TICKER'].unique()
    # DCN is the unique identification code for the reports

    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns = get_file_paths(dcns)
    return all_files_dcns



def get_all_companies(df, indexes):
    """
    Return the set of companies in the dataframe with the given indexes
    """
    all_companies = df.iloc[list(indexes), :]['TICKER'].unique()
    return all_companies


def get_file_paths(target_dcns):
    """
    Return a list of tuples that contains file paths and DCNs of all reports with the target DCNs
    """
    # directory = r".\PDFParsing\parsed_files"
    directory = r".\PDFParsing\clean_txt_flat"
    files = []
    temp = os.path.join(directory)
    list_files = os.listdir(temp)
    for item in list_files:
        l = item.split("-")
        dcn = l[-1].rstrip(".txt").rstrip("(1)")
        while dcn and not dcn[-1].isdigit():
            dcn = dcn[:-1]
        while dcn and not dcn[0].isdigit():
            dcn = dcn[1:]
        if dcn:
            dcn = int(dcn)
        else:
            continue
        if dcn in target_dcns:
            files.append((os.path.join(temp, item), dcn))
    return files

