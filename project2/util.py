import nltk
import string

from nltk.util import ngrams
from nltk.corpus import stopwords


def readDocument(docPathName):
    file = open(docPathName, "r")
    text = file.read().lower()
    return text

def removeStopWords(list_terms):
    return [token for token in list_terms if token not in stopwords.words('english')]

def removePunctuation(text):
    return text.translate(None, string.punctuation)

def getWordGrams(words, min=1, max=3):
    """
    Getting n-grams in a specified range
    """
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
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # top 5 candidates
    for candidate in top_candidates[:n]:
        print("" + str(candidate[0]) + " - " + str(candidate[1]))
