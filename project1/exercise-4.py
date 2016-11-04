import math
import os

from util import readDocument
from util import prepareDocuments
from util import getWordGrams
from util import printTopNTerms

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

def calcLM2(term, freq_dict, total_unigram_count):

    term_freq = 1   # add-1 smoothing

    if term in freq_dict:
        term_freq = freq_dict[term]

    return float(term_freq) / float(total_unigram_count)

##################################################################
## Main starts here
##################################################################

fg = prepareDocuments([readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex4"))])
bg = prepareDocuments(["palavras lol epa palavras lol epa assim lol epa lol lol epa assim doc1"])   #TODO read background from collection

#dictionaries holding each term (n-gram) ocurrance number
fg_dict, fg_unigram_count = buildTermCountDict(fg)
bg_dict, bg_unigram_count = buildTermCountDict(bg)

#KLDiv score per candidate
scores = calcCandidateKLDivScore(fg_dict, bg_dict, fg_unigram_count, bg_unigram_count)

printTopNTerms(scores, 5)
