from numpy import dot
import os
import nltk
import math
from nltk.tokenize.punkt import PunktSentenceTokenizer
from util import readDocument
from util import getWordGrams
from util import createGraph
from util import pagerank
from util import getCandidatesfromDocumentSentences
from util import getTopCandidates
from util import precision
from util import f1
from util import recall
from util import avg_precision
from util import mean_avg_precision
from util import calculateBM25Feature
from util import getDocumentRelevantKeyphrases
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import calculateDocumentEvaluation

from multiprocessing.dummy import Pool as ThreadPool

from time import gmtime, strftime


########################### PhrInf ###############################

def buildTermCountDict(doc_list):

    freq_dict = {}

    for doc in doc_list:
        n_grams = getWordGrams(doc, min=1, max=3)

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

    for term in fg_term_freq:

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



def checkKeyphrase(docName, term, training_document=False):
    #returns True if the term is a keyphrase of docName
    keyphrases = getDocumentRelevantKeyphrases(docName, training_document)
    if term in keyphrases:
        return True
    else:
        return False




def calculatePositionFeature(n_grammed_sentences):
    scores = {}
    number_sentences = len(n_grammed_sentences)
    for idx, sentence in enumerate(n_grammed_sentences):
        prior_weight = number_sentences - idx  # first sentences have better candidates
        for gram in sentence:
            if gram not in scores:
                scores[gram] = float(prior_weight)
    return scores

def calculatePRankFeature(n_grams, n_grammed_sentences):
    return pagerank(createGraph(n_grams, n_grammed_sentences))

def calculatePhrInfFeature(docName, n_grammed_docs):
    fg_dict, fg_unigram_count = buildTermCountDict([n_grammed_docs[docName]])
    bg_dict, bg_unigram_count = buildTermCountDict(n_grammed_docs.values())
    return calcCandidateKLDivScore(fg_dict, bg_dict, fg_unigram_count, bg_unigram_count)


def generateTrainingData():
    docNames = getDocumentNames(True)
    print "getAllDocumentCandidates: " + strftime("%H:%M:%S", gmtime())
    candidates_in_sentences = getAllDocumentCandidates(docNames, True)
    candidates = {}
    for doc in candidates_in_sentences:
        candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])
    training_data = []
    for docName in candidates:
        print "scoresPos: " + strftime("%H:%M:%S", gmtime())
        scoresPos = calculatePositionFeature(candidates_in_sentences[docName])
        print "scoresBM25: " + strftime("%H:%M:%S", gmtime())
        scoresBM25 = calculateBM25Feature(docName, candidates)
        print "scoresPhrInf: " + strftime("%H:%M:%S", gmtime())
        scoresPhrInf = calculatePhrInfFeature(docName, candidates)
        scoresPRank = calculatePRankFeature(candidates[docName], candidates_in_sentences[docName])
        for term in scoresPos:
            #create feature vector
            features = []
            features.append(scoresPos[term])
            features.append(scoresBM25[term])
            features.append(scoresPhrInf[term])
            features.append(scoresPRank[term])
            #check if term is keyphrase
            bool = checkKeyphrase(docName, term, True)
            training_data.append((features, bool))
    return training_data

def generateEvaluationData():
    docNames = getDocumentNames()
    candidates_in_sentences = getAllDocumentCandidates(docNames)
    candidates = {}
    for doc in candidates_in_sentences:
        candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])
    evaluation_data = {}
    for docName in candidates:
        evaluation_data[docName] = {}
        print "scoresPos: " + strftime("%H:%M:%S", gmtime())
        scoresPos = calculatePositionFeature(candidates_in_sentences[docName])
        print "scoresBM25: " + strftime("%H:%M:%S", gmtime())
        scoresBM25 = calculateBM25Feature(docName, candidates)
        print "scoresPhrInf: " + strftime("%H:%M:%S", gmtime())
        scoresPhrInf = calculatePhrInfFeature(docName, candidates)
        scoresPRank = calculatePRankFeature(candidates[docName], candidates_in_sentences[docName])
        for term in scoresPos:
            #create feature vector
            features = []
            features.append(scoresPos[term])
            features.append(scoresBM25[term])
            features.append(scoresPhrInf[term])
            features.append(scoresPRank[term])

            evaluation_data[docName][term] = features

    return evaluation_data

def Perceptron(training_data):
    # receives a list of tuples. Each tuple is composed of a list of floats and a boolean (list, boolean)
    # the list of floats contains the candidates' features and the boolean is True if the candidate is a keyword

    # returns the weight vector

    w = [0] * len(training_data[0][0]) #weight vector
    b = 0 #threshold

    for x in training_data:
        if (dot(w, x[0]) < b):
            yp = False
        else:
            yp = True

        y = x[1]

        if(yp != y):

            if(y==False):
                yr = -1
            else:
                yr = 1

            if(yr*(dot(w,x[0])-b) <= 0):
                for i in range(0, len(w)):
                    w[i] = w[i] + yr * x[0][i]
                b -= yr

    return w

def calculateDocumentScores(doc, w):
    scores = {}
    for term in doc:
        scores[term] = dot(w, doc[term])
    return scores

##################################################################
## Main starts here
##################################################################
print "Generating training data..."
training_data = generateTrainingData()
print "Training data generated!"
print "Running the Perceptron..."
weight_vector = Perceptron(training_data)
print "Generating evaluation data..."
evaluation_data = generateEvaluationData()
print "Evaluation data generated!\n"

map = []
for docName in evaluation_data:
    scores = calculateDocumentScores(evaluation_data[docName], weight_vector)
    values = calculateDocumentEvaluation(docName, scores)
    print "====== " + docName + " ======"
    print "Precision: " + str(values["precision"])
    print "Recall: " + str(values["recall"])
    print "F1: " + str(values["f1"]) + "\n"
    map.append(values["ap"])

print "==================="
print "Mean Av. Precision: " + str(mean_avg_precision(map))