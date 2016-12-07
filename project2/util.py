import nltk
import string
import re
from nltk.util import ngrams
from nltk.corpus import stopwords


def readDocument(docPathName):
    file = open(docPathName, "r")
    text = file.read().lower()
    return text

def removeStopWords(list_terms):
    return [token for token in list_terms if token not in stopwords.words('english')]

def removePunctuation(text):

    if type(text) is str:
        return text.translate(None, string.punctuation)
    else:   #unicode
        return text.translate(dict((ord(char), None) for char in string.punctuation))

def getWordGrams(words, min=1, max=3):
    """
    Getting n-grams in a specified range
    """

    if '' in words:       #just in case
        words.remove('')

    s = []
    remove = False

    for n in range(min, max):
        for ngram in ngrams(words, n):
            for idx, word in enumerate(ngram):
                if word in stopwords.words('english') and not (len(ngram) == 3 and idx == 1):
                    remove = True
                    break

            if remove == False:
                s.append(' '.join(ngram))
            else:
                remove = False


    return s

def printTopCandidates(scores, n):

    # reverse ordering of candidates scores
    top_candidates = getOrderedCandidates(scores)

    # top 5 candidates
    for candidate in top_candidates[:n]:
        print("" + str(candidate[0]) + " - " + str(candidate[1]))


def getOrderedCandidates(scores):   #decreasing order
    # reverse ordering of candidates scores
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def tokenizeDocument(document):
    """
    Tokenizes, removes punctuation

    :param document: string
    :return: list of strings (document terms)
    """

    translate_table = dict((ord(char), None) for char in string.punctuation)
    removed_punctuation = document.translate(translate_table)
    tokenized_docs = nltk.word_tokenize(removed_punctuation)

    return tokenized_docs

def prepareDocuments(documents):
    """
    Tokenizes, removes punctuation

    :param documents: list of strings (documents)
    :return: list of list of strings (each documents terms)
    """

    clean_docs = []

    for document in documents:
        translate_table = dict((ord(char), None) for char in string.punctuation)
        removed_punctuation = document.translate(translate_table)
        clean_docs.append(removed_punctuation)

    tokenized_docs = [nltk.word_tokenize(d) for d in clean_docs]

    return tokenized_docs