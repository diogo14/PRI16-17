import numpy
import os
from glob import glob
from pprint import pprint as pp

def precision(list_r, list_a):
    # intersection of R with A
    i = set(list_r).intersection(list_a)
    len2 = len(list_a)
    if len2 == 0:
        return 0
    else:
        # relevant docs(R intersection A) within all docs(A)
        return len(i) / len2


def recall(list_r, list_a):
    i = set(list_r).intersection(list_a)
    # the same but divided by R
    return len(i) / len(list_r)


def f1_score(list_r, list_a):
    # use above functions
    prec = precision(list_r, list_a)
    rec = recall(list_r, list_a)
    if prec + rec == 0:
        return 0
    else:
        return 2*(prec * rec)/(rec + prec)


def avg_precision(list1, list2, k=5):

    if len(list2) > k:
        list2 = list2[:k]

    score = 0.0
    hits = 0.0

    for i, p in enumerate(list2):
        if p in list1 and p not in list2[:i]:
            hits += 1.0
            score += hits / (i + 1.0)

    if not list1:
        return 0.0

    return score / min(len(list1), k)


def mean_avg_precision(list1, list2, k=5):
    return numpy.mean([avg_precision(l1, l2, k) for l1, l2 in zip(list1, list2)])

#-------------------
# NOT TESTED YET
#-------------------
# function document_words
def document_words(fileglob=os.path.join(os.path.dirname(__file__), os.pardir, "documents\\")):
#def document_words(docName, directory=os.path.join(os.path.dirname(__file__), os.pardir, "documents\\")):
        #split words and return a dictionary with document and list of words.
        # cycle for for all documents
    texts, words = {}, set()
    for txtfile in glob(fileglob):
        with open(txtfile, 'r') as f:
            txt = f.read().split()
            words |= set(txt)
            texts[txtfile.split('\\')[-1]] = txt
    return texts, words

texts, words = document_words()

#{'document_x': ['word1', 'word2' ...]}
pp(texts)
# same but sorted without duplicates
pp(sorted(words))


# function return distinct keywords
def distinct_keywords(fileglob=os.path.join(os.path.dirname(__file__), os.pardir, "indexers\\")):
        #retrieve content of each .key file for each document
    texts, words = {}, set()
    for txtfile in glob(fileglob):
        with open(txtfile, 'r') as f:
            txt = f.read().split()
            words |= set(txt)
            texts[txtfile.split('\\')[-1]] = txt
    return texts, words

#keywords in each indexer file
pp(sorted(words))
#missing check with all documents, maybe update the initial list of keywords.

#def document_keywords(document, words):