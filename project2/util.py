import nltk
import string
import codecs
from nltk.util import ngrams
from nltk.corpus import stopwords


def readDocument(path):
    file = codecs.open(path, "r", "utf-8")
    text = file.read().lower()
    return text

def writeToFile(path, content):
    f = open(path, 'w')
    f.write(content)
    f.close()


def getWordGrams(words, min=1, max=3):
    """ Getting n-grams in a specified range"""

    if '' in words:       #just in case
        words.remove('')

    s = []
    remove = False

    for n in range(min, max):
        for ngram in ngrams(words, n):
            for idx, word in enumerate(ngram):
                if len(word) == 1 or word.isnumeric() or word in stopwords.words('english') and not (len(ngram) == 3 and idx == 1):
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
        print("" + candidate[0] + " - " + str(candidate[1]))


def getOrderedCandidates(scores):   #decreasing order
    # reverse ordering of candidates scores
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
