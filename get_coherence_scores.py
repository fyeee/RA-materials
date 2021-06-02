import pandas as pd
import numpy as np
import math
import os
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import Powerset as ps
from gensim import corpora, models, similarities
import os
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import time
import warnings
import random
from gensim.models import CoherenceModel
warnings.filterwarnings("ignore", category=UserWarning)

# functions to organize data before building matrix
# -------------------------------------------------
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
def get_company_files(target_dcns):
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


# function to get a list of coherence scores
def get_coherence(df, industry, quarter):
    # dictionary: {Key=Industry Code, Value=Index of Report in Metadata}
    industries_to_index = construct_indu_index_mapping(df)

    # dictionary: {Key = Quarter 'YYYY qQ', Value = Index of Report in Metadata}
    quarter_to_index = construct_quar_index_mapping(df)

    # select all report indices (rows in metadata) for the industry-quarter
    indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
    # set of all company names for the industry-quarter
    all_companies = df.iloc[list(indexes), :].groupby('TICKER')["DCN"].count().reset_index()['TICKER'].tolist()
    # DCN is the unique identification code for the reports

    # subset_companies = ["AAL.OQ", 'ALK.N', 'FDX.N', "DAL.N", "UAL.OQ"]
    dcns = set(df.iloc[list(indexes), :]["DCN"])
    all_files_dcns = get_company_files(dcns)
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

    dictionary_LDA = corpora.Dictionary(words)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in words]

    coherence_values = []
    model_list = []

    for num_topics in range(1, 26, 1):
        print(num_topics)

        # generate the LDA Model
        lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary_LDA,
                                    num_topics=num_topics,
                                    random_state=100,
                                    chunksize=10,
                                    passes=100,
                                    alpha='auto',
                                    eta='auto')

        model_list.append(lda_model)

        coherencemodel = CoherenceModel(model=lda_model, dictionary=dictionary_LDA,
                                        texts=words, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return [model_list, coherence_values]


if __name__ == "__main__":
    start_time =  time.time()

    # load meta-data file
    df = pd.read_csv("cleaned_metadata_reports_noduplicates_with_industry.csv")

    df['industry'] = df['ggroup']
    df['quarter_year'] = df["year"].astype("str") + " q" + df["quarter"].astype("str")

    df2 = df.dropna(subset=['industry']).reset_index(drop=True)
    df2['industry-quarter'] = list(zip(df2.industry, df2.quarter_year))
    list_industries_quarter = df2.groupby('industry-quarter').count().reset_index()['industry-quarter'].tolist()

    #industry = 2030
    #quarter = '2018 q4'

    df_coherence=pd.DataFrame(columns=['Industry','Quarter','Coherence scores','Optimal topics'])

    list_dfs=[]
    iterloop=1
    for iq in list_industries_quarter:
        print(iq,iterloop)
        loop_time = time.time()

        industry=int(iq[0])
        quarter=iq[1]

        coherence_scores = get_coherence(df, industry, quarter)[1]
        optimal_topics=np.argmax(coherence_scores)

        df_coherence=df_coherence.append([industry,quarter,coherence_scores,optimal_topics])
        print(df_coherence.head())
        df_coherence.to_csv('industry_coherence_analysis.csv')
        print("Loop: --- %s seconds ---" % (time.time() - loop_time))




    print("Total: --- %s seconds ---" % (time.time() - start_time))


    df_coherence.to_csv('industry_coherence_analysis.csv')
