from gensim import corpora, models, similarities
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.colors as mcolors
from gensim.models import CoherenceModel

from nltk.tokenize import word_tokenize
from metadata_processing_helper import *


def construct_corpus(all_files_dcns):
    words = []
    did = 0
    for fname, _ in all_files_dcns:
        f = open(fname, 'r')
        result = f.readlines()
        tokens = []
        for i, line in enumerate(result):
            if "redistribut reproduct prohibit without written permiss copyright cfra document intend provid person invest advic take account specif invest" in line \
                    or "redistribut reproduct prohibit without prior written permiss copyright cfra" in line \
                    or "object financi situat particular need specif person may receiv report investor seek independ financi advic regard suitabl and/or appropri make" in line \
                    or "invest implement invest strategi discuss document understand statement regard futur prospect may realiz investor note incom" in line \
                    or "invest may fluctuat valu invest may rise fall accordingli investor may receiv back less origin invest investor seek advic concern impact" in line \
                    or "invest may person tax posit tax advisor pleas note public date document may contain specif inform longer current use make" in line \
                    or "invest decis unless otherwis indic intent updat document" in line:
                continue

            if "mm" not in line and len(word_tokenize(line)) > 2:
                tokens.extend(word_tokenize(line))
        #         tokens = word_tokenize(result)
        tokens = list(filter(("--").__ne__, tokens))
        tokens = list(filter(("fy").__ne__, tokens))
        tokens = list(filter(("cfra").__ne__, tokens))
        tokens = list(filter(("et").__ne__, tokens))
        tokens = list(filter(("am").__ne__, tokens))
        tokens = list(filter(("pm").__ne__, tokens))
        if "bankdatesecur" in tokens:
            continue
        words.append(tokens)
        did += 1

    dictionary_LDA = corpora.Dictionary(words)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in words]
    return words, dictionary_LDA, corpus


def get_coherence(all_files_dcns):
    # select all report indices (rows in metadata) for the industry-quarter
    # indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])

    ## Estimate LDA model for the entire industry (all companies)
    words, dictionary_LDA, corpus = construct_corpus(all_files_dcns)

    coherence_values = []
    model_list = []

    for num_topics in range(3, 20):
        print(num_topics)
        # generate the LDA Model
        lda_time = time.time()
        lda_model = models.LdaModel(corpus=corpus,
                                        id2word=dictionary_LDA,
                                        num_topics=num_topics,
                                        random_state=100,
                                        chunksize=50,
                                        passes=50,
                                        alpha='auto',
                                        eta=0.01)
        print("LDA: --- %s seconds ---" % (time.time() - lda_time))

        model_list.append(lda_model)
        coherence_time = time.time()
        coherencemodel = CoherenceModel(model=lda_model, dictionary=dictionary_LDA,
                                        texts=words, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("Coherence: --- %s seconds ---" % (time.time() - coherence_time))

    return [model_list, coherence_values]


# get a factor loading matrix for each industry + analyst names
def get_topic_industry(all_files_dcns, num_topics):

    ## Estimate LDA model for the entire industry (all companies)
    words, dictionary_LDA, corpus = construct_corpus(all_files_dcns)

    # generate the LDA Model
    lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary_LDA,
                                    num_topics=num_topics,
                                    random_state=100,
                                    chunksize=50,
                                    passes=100,
                                    alpha='auto',
                                    eta=0.01)

    # set alpha='auto' and eta='auto', such that the model learns from the data?

    return [lda_model, words, dictionary_LDA]

#
# # get a factor loading matrix for each stock + analyst names
# def get_factor_matrix(df, industry, quarter, lda_model, words, dictionary_LDA):
#     num_topics = len(lda_model.show_topics())
#
#     # select all report indices (rows in metadata) for the industry-quarter
#     indexes = industries_to_index[industry].intersection(quarter_to_index[quarter])
#     # set of all company names for the industry-quarter
#     all_companies = df.iloc[list(indexes), :]['TICKER'].unique()
#     # DCN is the unique identification code for the reports
#
#     dcns = set(df.iloc[list(indexes), :]["DCN"])
#     all_files_dcns = get_file_paths(dcns)
#     # dictionary: {Key=Analyst Name, Value = Index of Report in Metadata}
#     analyst_to_index = construct_analyst_index_mapping(df, all_files_dcns)
#
#     # set alpha='auto' and eta='auto', such that the model learns from the data?
#
#     loading_matrices = []
#     for companies in all_companies:
#         # print(companies)
#         dcns = set(df.iloc[list(indexes), :][df.TICKER == companies]["DCN"])
#         dcn_company = get_file_paths(dcns)
#         # print (dcn_company)
#         analyst_to_index = construct_analyst_index_mapping(df, dcn_company)
#         matrix = []
#         for analyst, anal_indexes in analyst_to_index.items():
#             row = [0] * num_topics
#             all_words = []
#             for i in anal_indexes:
#                 all_words.extend(words[i])
#             topics = lda_model.get_document_topics(dictionary_LDA.doc2bow(all_words), minimum_probability=1e-4)
#             for index, dist in topics:
#                 row[index] = dist
#             matrix.append(row)
#         matrix = np.array(matrix)
#         loading_matrices.append((companies, matrix, analyst_to_index))
#
#     return [loading_matrices, industry, quarter]
