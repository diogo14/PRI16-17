import os
import math
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

def readDocument(docName, directory=os.path.join(os.path.dirname(__file__), os.pardir, "resources\\")):
    """
    Reads a given document from 'resources' directory by default
    """

    file = open(directory + docName, "r")
    text = file.read().lower()
    no_punctuation = text #TODO remove punctuation

    return no_punctuation

def buildInvertedIndexDict(doc_list):
    """
    Creates a dictionary that for each term(n-gram) contains a dictionary of documents where that term occurrs and the
    number of occurrences. e.g., {'word' : {'document1' : 4}

    :param doc_list:  list of documents, each document is a list of terms(n-grams) that it containss
    :return: dictionary
    """

    total_term_number = 0
    inverted_index_dict = {}

    for term_list in doc_list:
        total_term_number += len(term_list)

        for term in term_list:
            if term in inverted_index_dict:
                if doc in inverted_index_dict[term]:
                    inverted_index_dict[term][doc] = inverted_index_dict[term][doc] + 1
                else:
                    inverted_index_dict[term][doc] = 1

            else:
                invertedIndexDict[term] = {doc : 1}

    print("# Terms: " + str(total_term_number))

    return inverted_index_dict

def calcTermDF(inverted_index_dict, term, document):
    """
    Returns the f(t, D) - frequcny for term t in document D
    """
    if term in inverted_index_dict:
        return inverted_index_dict[term][document]
    else:
        return 0

def calcTermIDF(inverted_index_dict, term, total_doc_number):
    """
    IDF of a given term

    IDF = log((N - n(t) + 0.5) / (n(t) + 0.5
    N - total number of documents in a background collection
    n(t) - number of documents, from this background, containing the term t
    """

    df = calcTermDF(inverted_index_dict, term)
    return math.log((total_doc_number - df + 0.5) / (df + 0.5))

def calcTermScore(inverted_index_dict, term, document, document_length, average_document_length, total_document_number):
    """
    Gets  score to a given candidate (term) within a given document according to the BM25 term weighting heuristic
    """

    k1 = 1.2
    b = 0.75

    idf = calcTermIDF(inverted_index_dict, term, total_document_number)
    df = calcTermDF(inverted_index_dict, term, document)

    return idf * (df * (k1 + 1)) / (df + k1 * (1 - b + b * document_length / average_document_length))

def getWordGrams(words, min=1, max=4):
    """
    Getting n-grams in a specified range
    """

    s = []

    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))

    return s

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
        print(tagged)
        print(observed_pattern)

        if(re.match(regex_pattern, observed_pattern)):
            filtered.append(candidate)

    return filtered

def removeStopWords(list_terms):
    return [token for token in list_terms if token not in stopwords.words('english')]


#Main starts here

#training data
train = fetch_20newsgroups(subset='train')

documents = [nltk.word_tokenize(d) for d in ["um dois tres", "quatro cinco seis"]]  #change to train.data as background collection
filtered_documents = [removeStopWords(d) for d in documents]
#TODO remove punctuation
n_grammed_documents = [getWordGrams(words) for words in filtered_documents]
print(n_grammed_documents)

filtering_regex = r""   #TODO choose good pattern to match
filtered_n_grams = [filterNGrams(n_gram, filtering_regex) for n_gram in n_grammed_documents]
print(filtered_n_grams)


#to be continued...