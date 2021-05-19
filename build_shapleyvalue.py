import pandas as pd
import numpy as np
import math
import os
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim import corpora, models, similarities
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def construct_indu_index_mapping(df):
    """
    Construct a dictionary with
    key: industry code
    value: indexes of all reports in the dataframe
    """
    industries_to_index = {}
    industries = df["ggroup"].dropna().astype(int).unique()
    industries = industries.tolist()
    quarters = (df["year"].astype("str") + " q" + df["quarter"].astype("str")).unique()
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
    quarters = (df["year"].astype("str") + " q" + df["quarter"].astype("str")).unique()
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
def get_all_companies(df, indexes):
    """
    Return the set of companies in the dataframe with the given indexes
    """
    raw_companies = df.iloc[list(indexes), 4].unique()
    all_companies = set()
    for item in raw_companies:
        l = item.split(",")
        for company in l:
            all_companies.add(company.strip(" ").strip("^L19"))
    return all_companies
def get_company_files(target_dcns, company):
    """
    Return a list of tuples that contains file paths and DCNs of all reports with the target DCNs
    """
    directory = r".\PDFParsing\parsed_files"
    files = []
    temp = os.path.join(directory, company)
    list_files = os.listdir(temp)
    for item in list_files:
        l = item.split("-")
        dcn = l[-1].rstrip(".txt")
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

def main():

    # load meta-data file
    df = pd.read_csv("metadata_reports_noduplicates_with_industry.csv")

    # dictionary: {Key=Industry Code, Value=Index of Report in Metadata}
    industries_to_index = construct_indu_index_mapping(df)

    # dictionary: {Key = Quarter 'YYYY qQ', Value = Index of Report in Metadata}
    quarter_to_index = construct_quar_index_mapping(df)

    # Define a given industry and index
    industry = 2030
    quarter = '2018 q4'

    # select all report indices (rows in metadata) for the industry-quarter
    indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
    # set of all company names for the industry-quarter
    all_companies = get_all_companies(df, indexes)
    # DCN is the unique identification code for the reports
    dcns = set(df.iloc[list(indexes), :]["DCN"])

    subset_companies = ["AAL.OQ", 'ALK.N', 'FDX.N', "DAL.N", "UAL.OQ"]
    all_files_dcns = []
    for companies in subset_companies:
        all_files_dcns += get_company_files(dcns, companies)
    # dictionary: {Key=Analyst Name, Value = Index of Report in Metadata}
    analyst_to_index = construct_analyst_index_mapping(df, all_files_dcns)

    ## Estimate LDA model for the entire industry (all companies)
    words = []
    did = 0
    for fname, _ in all_files_dcns:
        f = open(fname, 'r')
        result = f.read()
        tokens = word_tokenize(result)
        tokens = list(filter(("--").__ne__, tokens))
        tokens = list(filter(("fy").__ne__, tokens))
        words.append(tokens)
        did += 1
    # select number of topics
    num_topics = 8
    dictionary_LDA = corpora.Dictionary(words)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in words]

    # generate the LDA Model
    lda_model = models.LdaMulticore(corpus=corpus,
                                    id2word=dictionary_LDA,
                                    num_topics=num_topics,
                                    random_state=100,
                                    chunksize=10,
                                    passes=10,
                                    alpha=0.3,
                                    eta=0.6)
    loading_matrices = []
    for companies in subset_companies:
        dcn_company=get_company_files(dcns, companies)
        analyst_to_index = construct_analyst_index_mapping(df, dcn_company)
        matrix=[]
        for analyst, indexes in analyst_to_index.items():
            row = [0] * num_topics
            all_words = []
            for i in indexes:
                all_words.extend(words[i])
            topics = lda_model.get_document_topics(dictionary_LDA.doc2bow(all_words), minimum_probability=1e-4)
            for index, dist in topics:
                row[index] = dist
            matrix.append(row)
        matrix = np.array(matrix)
        loading_matrices.append((companies,matrix,analyst_to_index))

    return loading_matrices

print(__name__)

if __name__=="__main__":
    print("a")
    loading_matrices=main()

