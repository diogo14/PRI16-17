import os
import math
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

def readDocument(docPathName=os.path.join(os.path.dirname(__file__), "resources", "doc_ex1")):
    """
    Reads a given document from 'resources' directory by default
    """
    file = open(docPathName, "r")
    text = file.read().lower()
    #no_punctuation = text #TODO remove punctuation

    return text

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
    return math.log(total_doc_number / n_t)

def performCandidateScoring(inverted_index_dict, candidates, number_background_documents):
    scores = {}

    for candidate in candidates:
        if candidate in scores or candidate not in inverted_index_dict:
            continue

        tf = candidates.count(candidate)    #candidate frequency within document being analyzed
        idf = calcCandidateIDF(inverted_index_dict, candidate, number_background_documents)
        scores[candidate] = tf * idf

        #print("score('" + candidate + "') = " + str(scores[candidate]))


    return scores

def getWordGrams(words, min=1, max=3):
    """
    Getting n-grams in a specified range
    """

    s = []

    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))

    return s

def removeStopWords(list_terms):
    return [token for token in list_terms if token not in stopwords.words('english')]

def prepareDocuments(documents):
    """
    Tokenizes, removes stopwords and punctuation

    :param documents: list of strings (documents)
    :return: list of list of strings (each documents terms)
    """
    prepared = [nltk.word_tokenize(d) for d in documents]
    prepared = [removeStopWords(d) for d in prepared]
    # TODO remove punctuation

    return prepared


##################################################################
## Main starts here
##################################################################

#get foreground document
foreground_document = readDocument()

#get training document set
training_dataset = fetch_20newsgroups(subset='train').data

#create background document set
background_documents = training_dataset + [foreground_document]

#tokenizing, removing stopwords and punctuation from background collection
#documents = prepareDocuments(["doc1 words here", "doc2 words here example example"])

#getting background documents n-grams
#bi_grammed_background_documents = [getWordGrams(words) for words in documents]

#building structure that holds background candidate occurances over documents
#inverted_index_dict = buildInvertedIndexDict(bi_grammed_background_documents)


#document to analyze
#document = readDocument("doc_ex1")
#query_document_terms = prepareDocuments([document])

#n-grammed document
#bi_grammed_document = [getWordGrams(words) for words in query_document_terms]

#score for each candidate
#scores = performCandidateScoring(inverted_index_dict, bi_grammed_document[0], len(background_documents))

#reverse ordering of candidates scores
#top_candidates = sorted(scores.items(), key = lambda x: x[1], reverse = True)

#top 5 candidates
#for candidate in top_candidates[:5]:
 #   print("" + str(candidate[0]) + " - " + str(candidate[1]))


