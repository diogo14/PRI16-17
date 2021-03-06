import os
import math
import re
import nltk
import datetime

from sklearn.datasets import fetch_20newsgroups


from util import readDocument
from util import prepareDocuments
from util import getWordGrams
from util import printTopNTerms

def buildInvertedIndexDict(documents_list):
    """
    Creates a dictionary that for each term(n-gram) contains a dictionary of documents where that term occurrs and the
    number of occurrences. e.g., {'word' : {'document1' : 4}

    :param doc_list:  list of documents, each document is a list of terms(n-grams) that it containss
    :return: dictionary
    """

    total_term_number = 0
    inverted_index_dict = {}

    for doc_index, doc_term_list in enumerate(documents_list):
        total_term_number += len(doc_term_list)

        for term in doc_term_list:
            if term in inverted_index_dict:
                if doc_index in inverted_index_dict[term]:
                    inverted_index_dict[term][doc_index] = inverted_index_dict[term][doc_index] + 1
                else:
                    inverted_index_dict[term][doc_index] = 1

            else:
                inverted_index_dict[term] = {doc_index : 1}

    return inverted_index_dict

def calcCandidateIDF(inverted_index_dict, candidate, total_doc_number):
    """
    IDF of a given candidate

    IDF = log((N - n(t) + 0.5) / (n(t) + 0.5
    N - total number of documents in a background collection
    n(t) - number of documents, from this background, containing the term t
    """

    n_t = len(inverted_index_dict[candidate])
    return math.log((total_doc_number - n_t + 0.5) / (n_t + 0.5))

def calcCandidateScore(inverted_index_dict, n_grammed_document, candidate, document_length, average_document_length, number_background_documents):
    """
    Gets  score to a given candidate  within a given document according to the BM25 term weighting heuristic
    """

    k1 = 1.2
    b = 0.75

    idf = calcCandidateIDF(inverted_index_dict, candidate, number_background_documents)

    #f(t, D) - frequency for candidate t in document D
    tf = n_grammed_document.count(candidate)

    return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (document_length / average_document_length))))

def performCandidateScoring(inverted_index_dict, candidates, n_grammed_document, average_document_length, number_background_documents):
    scores = {}

    document_length = len(n_grammed_document)

    for candidate in set(candidates):

        if candidate not in inverted_index_dict:
            continue

        candidate_doc_score = calcCandidateScore(inverted_index_dict, n_grammed_document, candidate, document_length,
                                                 average_document_length, number_background_documents)

        #print("score('" + candidate + "') = " + str(candidate_doc_score))

        if candidate not in scores:
            scores[candidate] = candidate_doc_score

    return scores

def filterNGrams(list, regex_pattern):
    """
    Return a list of n-grams after filtering with a given REGEX
    """

    filtered = []

    for candidate in list:
        tagged = nltk.pos_tag(nltk.word_tokenize(candidate))
        observed_pattern = ""
        for pair in tagged:
            observed_pattern += " " + pair[1]
        #print(tagged)
        #print(observed_pattern)

        if(re.match(regex_pattern, observed_pattern)):
            filtered.append(candidate)

    return filtered

def retrieveAverageDocLength(documents):
    total_documents_terms = 0

    for doc_term_list in documents:
        total_documents_terms += len(doc_term_list)

    return total_documents_terms / len(documents)


##################################################################
## Main starts here
##################################################################

#background documents
background_documents = fetch_20newsgroups(subset='train')

print("Preparing background " + str(datetime.datetime.utcnow()))
#tokenizing, removing punctuation from background collection
documents = prepareDocuments(background_documents.data)

#getting background  avgdl and N
average_document_length = retrieveAverageDocLength(documents)
number_background_documents = len(documents)

print("Getting n-grams " + str(datetime.datetime.utcnow()))
#getting background documents n-grams
n_grammed_background_documents = [getWordGrams(words, min=1, max=4) for words in documents]

print("Filtering candidates " + str(datetime.datetime.utcnow()))
#filtering background candidates
filtering_regex = r"((JJ\s)*(NN(S|(PS)|P)?\s)+IN\s)?((JJ\s)*(NN(S|(PS)|P)?\s?)+)+"
filtered_background_n_grams = [filterNGrams(n_gram, filtering_regex) for n_gram in n_grammed_background_documents]

print("Building occurance dictionary " + str(datetime.datetime.utcnow()))
#building structure that holds background candidate occurances over documents
inverted_index_dict = buildInvertedIndexDict(filtered_background_n_grams)

print("#### Foreground ####")

print("Preparing " + str(datetime.datetime.utcnow()))
#document to analyze
query_document_terms = prepareDocuments([unicode(readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex3")), "ISO-8859-1")])

print("Getting n-grams " + str(datetime.datetime.utcnow()))
#n-grammed document
n_grammed_document = [getWordGrams(words, min=1, max=4) for words in query_document_terms]

print("Filtering " + str(datetime.datetime.utcnow()))
#filtering
filtered_n_grams = [filterNGrams(n_gram, filtering_regex) for n_gram in n_grammed_document]

print("Scoring candidates " + str(datetime.datetime.utcnow()))
#score for each candidate
scores = performCandidateScoring(inverted_index_dict, filtered_n_grams[0], n_grammed_document[0], average_document_length, number_background_documents)

printTopNTerms(scores, 5)

