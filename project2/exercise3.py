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

from multiprocessing.dummy import Pool as ThreadPool

from time import gmtime, strftime


############################ BM25 ################################
def retrieveAverageDocLength(documents):
    total_documents_terms = 0

    for doc_term_list in documents:
        total_documents_terms += len(doc_term_list)

    return total_documents_terms / len(documents)


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

def performCandidateScoring(inverted_index_dict, n_grammed_document, average_document_length, number_background_documents):
    scores = {}

    document_length = len(n_grammed_document)

    for candidate in set(n_grammed_document):

        if candidate not in inverted_index_dict:
            continue

        candidate_doc_score = calcCandidateScore(inverted_index_dict, n_grammed_document, candidate, document_length,
                                                 average_document_length, number_background_documents)

        #print("score('" + candidate + "') = " + str(candidate_doc_score))

        if candidate not in scores:
            scores[candidate] = candidate_doc_score

    return scores

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

def getKeyphrasesFromFile(filePathName):
    keyPhrases = readDocument(filePathName).splitlines()
    return keyPhrases

def checkKeyphrase(docName, term, training_document=False):
    #returns True if the term is a keyphrase of docName
    keyphrases = getDocumentRelevantKeyphrases(docName, training_document)
    if term in keyphrases:
        return True
    else:
        return False


def getDocumentNames(training_document=False):
    if (training_document == False):
        path = os.path.join(os.path.dirname(__file__), "dataset", "documents")
    else:
        path = os.path.join(os.path.dirname(__file__), "training", "documents")
    fileNames = os.listdir(path)
    fileNames.sort()
    return fileNames

def getDocumentContent(docName, training_document=False):
    if (training_document == False):
        path = os.path.join(os.path.dirname(__file__), "dataset", "documents", docName)
    else:
        path = os.path.join(os.path.dirname(__file__), "training", "documents", docName)
    return readDocument(path)

def getDocumentRelevantKeyphrases(docName, training_document=False):
    if(training_document == False):
        rootPath = os.path.join(os.path.dirname(__file__), "dataset", "indexers")
        k = []
        for i in range(1, 7):
            k.append(getKeyphrasesFromFile(os.path.join(rootPath, "iic" + str(i), docName[:-3] + "key")))
        return list(set().union(k[0], k[1], k[2], k[3], k[4], k[5]))
    else:
        rootPath = os.path.join(os.path.dirname(__file__), "training", "keys")
        return getKeyphrasesFromFile(os.path.join(rootPath, docName[:-3] + "key"))

def getDocumentCandidates(docName, training_document=False):
    #returns a list of list of strings (each list contains the ngrams of a sentece)
    text = getDocumentContent(docName, training_document)
    sentences = PunktSentenceTokenizer().tokenize(text)
    return [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

def getAllDocumentCandidates(docNames, training_documents=False):
    #returns a dictionary of docNames to lists of lists (docs - sentences - terms)
    allCandidates = {}
    for docName in docNames[:2]:
        allCandidates[docName] = getDocumentCandidates(docName, training_documents)

    return allCandidates

def calculatePositionFeature(n_grammed_sentences):
    scores = {}
    number_sentences = len(n_grammed_sentences)
    for idx, sentence in enumerate(n_grammed_sentences):
        prior_weight = number_sentences - idx  # first sentences have better candidates
        for gram in sentence:
            if gram not in scores:
                scores[gram] = float(prior_weight)
    return scores

def calculatePRankFeature(n_grammed_sentences):
    return pagerank(createGraph(n_grammed_sentences))

def calculateBM25Feature(FGn_grammed_sentences, BGn_grammed_docs):
    FGngrams  = getCandidatesfromDocumentSentences(FGn_grammed_sentences)
    BGngrams = []
    for BGdoc in BGn_grammed_docs:
        BGngrams.append(getCandidatesfromDocumentSentences(BGn_grammed_docs[BGdoc]))
    number_background_documents = len(BGngrams)
    average_document_length = retrieveAverageDocLength(BGngrams)
    inverted_index_dict = buildInvertedIndexDict(BGngrams)
    return performCandidateScoring(inverted_index_dict, FGngrams, average_document_length,
                            number_background_documents)

def calculatePhrInfFeature(FGn_grammed_sentences, BGn_grammed_docs):
    FGngrams = getCandidatesfromDocumentSentences(FGn_grammed_sentences)
    BGngrams = []
    for BGdoc in BGn_grammed_docs:
        BGngrams.append(getCandidatesfromDocumentSentences(BGn_grammed_docs[BGdoc]))
    fg_dict, fg_unigram_count = buildTermCountDict([FGngrams])
    bg_dict, bg_unigram_count = buildTermCountDict(BGngrams)
    return calcCandidateKLDivScore(fg_dict, bg_dict, fg_unigram_count, bg_unigram_count)


def generateTrainingData():
    docNames = getDocumentNames(True)
    print "getAllDocumentCandidates: " + strftime("%H:%M:%S", gmtime())
    candidates = getAllDocumentCandidates(docNames, True)
    training_data = []
    for docName in candidates:
        print "scoresPos: " + strftime("%H:%M:%S", gmtime())
        scoresPos = calculatePositionFeature(candidates[docName])
        print "scoresBM25: " + strftime("%H:%M:%S", gmtime())
        scoresBM25 = calculateBM25Feature(candidates[docName], candidates)
        print "scoresPhrInf: " + strftime("%H:%M:%S", gmtime())
        scoresPhrInf = calculatePhrInfFeature(candidates[docName], candidates)
        #scoresPRank = calculatePRankFeature(candidates[docName])
        for term in scoresPos:
            #create feature vector
            features = []
            features.append(scoresPos[term])
            features.append(scoresBM25[term])
            features.append(scoresPhrInf[term])
            #features.append(scoresPRank[term])
            #check if term is keyphrase
            bool = checkKeyphrase(docName, term, True)
            training_data.append((features, bool))
    return training_data

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

def generateEvaluationData():
    docNames = getDocumentNames()
    candidates = getAllDocumentCandidates(docNames)

    evaluation_data = {}
    for docName in candidates:
        evaluation_data[docName] = {}
        scoresPos = calculatePositionFeature(candidates[docName])
        scoresBM25 = calculateBM25Feature(candidates[docName], candidates)
        scoresPhrInf = calculatePhrInfFeature(candidates[docName], candidates)
        #scoresPRank = calculatePRankFeature(candidates[docName])
        for term in scoresPos:
            #create feature vector
            features = []
            features.append(scoresPos[term])
            features.append(scoresBM25[term])
            features.append(scoresPhrInf[term])
            #features.append(scoresPRank[term])

            evaluation_data[docName][term] = features

    return evaluation_data




def calculateDocumentScores(doc, w):
    scores = {}
    for term in doc:
        scores[term] = dot(w, doc[term])
    return scores

# calculates the various evaluation values for a given document
def calculateDocumentEvaluation(docName, termsFeatures, w):
    scores = calculateDocumentScores(termsFeatures, w)
    retrieved = getTopCandidates(scores, 5)
    relevant = getDocumentRelevantKeyphrases(docName)
    values = {}
    values["precision"] = precision(relevant, retrieved)
    values["recall"] = recall(relevant, retrieved)
    values["f1"] = f1(values["precision"], values["recall"])
    values["ap"] = avg_precision(relevant, retrieved)
    return values

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
    values = calculateDocumentEvaluation(docName, evaluation_data[docName], weight_vector)
    print "====== " + docName + " ======"
    print "Precision: " + str(values["precision"])
    print "Recall: " + str(values["recall"])
    print "F1: " + str(values["f1"]) + "\n"
    map.append(values["ap"])

print "==================="
print "Mean Av. Precision: " + str(mean_avg_precision(map))