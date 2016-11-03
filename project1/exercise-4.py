import os
import math
import operator
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

def readDocument(docName, directory=os.path.join(os.path.dirname(__file__), os.pardir, "resources\\")):
    """
    Reads a given document from 'resources' directory by default
    """

    file = open(directory + docName, "r")
    text = file.read().lower()
    no_punctuation = text #TODO remove punctuation

    return no_punctuation

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

def removeStopWords(list_terms):
    return [token for token in list_terms if token not in stopwords.words('english')]

def getWordGrams(words, min=1, max=3):
    """
    Getting n-grams in a specified range
    """

    s = []

    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))

    return s

def buildTermCountDict(doc_list):

    freq_dict = {}

    for doc in doc_list:
        n_grams = getWordGrams(doc)

        for term in n_grams:
            if term in freq_dict:
                freq_dict[term] = freq_dict[term] + 1
            else:
                freq_dict[term] = 1

    total_unigram_count = 0

    for term in freq_dict:
        if ' ' not in term:
            total_unigram_count += freq_dict[term]

    return freq_dict, total_unigram_count

def calcCandidateKLDivScore(fg_term_freq, bg_term_freq, fg_unigram_count, bg_unigram_count):
    """
    Calculate KLDiv per term (unigram and bigram) from foreground document

    :param foreground_term_freq: dictionary holding frequency of each term in foreground document
    :return:  dictionary with score ( KLDiv = KlDivInformativeness + KlDivPhraseness ) per term
    """

    kldiv_per_term = {}

    for term in fg_dict:

        fg_unigram_term_probability = calcLM1(term, fg_term_freq, fg_unigram_count)
        bg_unigram_term_probability = calcLM1(term, bg_term_freq, bg_unigram_count)
        bg_bigram_term_probability = calcLM2(term, bg_term_freq, bg_unigram_count)

        klDivInformativeness = fg_unigram_term_probability * math.log(fg_unigram_term_probability / bg_unigram_term_probability)
        klDivPhraseness = bg_bigram_term_probability * math.log(bg_bigram_term_probability / bg_unigram_term_probability)

        klDiv = klDivInformativeness + klDivPhraseness

        kldiv_per_term[term] = klDiv

    return kldiv_per_term

def calcLM1(term, freq_dict, total_unigram_count):

    unigrams = term.split(" ")
    relfreq_unigrams = 1.0

    for unigram in unigrams:

        u_freq = 1  # add-1 smoothing

        if unigram in freq_dict:
            u_freq = freq_dict[unigram]

        u_relfreq = float(u_freq) / float(total_unigram_count)
        relfreq_unigrams *= u_relfreq

    return relfreq_unigrams

def calcLM2(term, freq_dict, total_unigram_count):  #TODO

    term_freq = 1   # add-1 smoothing

    if term in freq_dict:
        term_freq = freq_dict[term]

    return float(term_freq) / float(total_unigram_count)

def printTopNTerms(scores, n):

    # reverse ordering of candidates scores
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # top 5 candidates
    for candidate in top_candidates[:n]:
        print("" + str(candidate[0]) + " - " + str(candidate[1]))


##################################################################
## Main starts here
##################################################################

#building structure (dict) to hold each term (n-gram) ocurrance number
#for both foreground and background corpus

fg = prepareDocuments([readDocument("doc_ex4")])
bg = prepareDocuments(["palavras lol epa palavras lol epa assim lol epa lol lol epa assim doc1"])

fg_dict, fg_unigram_count = buildTermCountDict(fg)
bg_dict, bg_unigram_count = buildTermCountDict(bg) #TODO background collection

#KLDiv score per candidate
scores = calcCandidateKLDivScore(fg_dict, bg_dict, fg_unigram_count, bg_unigram_count)

printTopNTerms(scores, 5)
