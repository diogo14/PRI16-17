import nltk
import string
import codecs
from nltk.util import ngrams
from nltk.corpus import stopwords
import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import math
import re

from multiprocessing.dummy import Pool as ThreadPool

from time import gmtime, strftime

def readDocument(docPathName):
    file = codecs.open(docPathName, "r", "ISO-8859-1")
    text = file.read().lower()
    return text

def getWordGrams(words, min=1, max=3):
    """ Getting n-grams in a specified range"""

    if '' in words:       #just in case
        words.remove('')

    s = []
    remove = False

    for n in range(min, max):
        for ngram in ngrams(words, n):
            for idx, word in enumerate(ngram):
                has_letter = bool(re.search(r"[^\W\d_]", word, re.UNICODE))
                # bool(re.match(r"(\w|\u2014|\d){2,}", word, re.UNICODE)) and
                if not has_letter or word in stopwords.words('english') and not (len(ngram) == 3 and idx == 1):
                    remove = True
                    break

            if remove == False:
                s.append(' '.join(ngram))
            else:
                remove = False
    return s

def getTopCandidates(scores, n):
    # reverse ordering of candidates scores
    top_candidates = getOrderedCandidates(scores)
    return top_candidates[:n]


def printTopCandidates(scores, n):
    # top 5 candidates
    for candidate in getTopCandidates(scores, n):
        print("" + candidate[0] + " - " + str(candidate[1]))


def getOrderedCandidates(scores):   #decreasing order
    # reverse ordering of candidates scores
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def writeToFile(path, content):
    f = open(path, 'w')
    f.write(content)
    f.close()

def pagerank(graph):
    """Calculates PageRank for an undirected graph"""

    damping = 0.85
    N = graph.number_of_nodes()  # number of candidates
    convergence_threshold = 0.0001

    scores = dict.fromkeys(graph.nodes(), 1.0 / N)  #initial value

    for _ in xrange(100):
        convergences_achieved = 0
        for candidate in graph.nodes():
            linked_candidates = graph.neighbors(candidate)
            rank = (1-damping)/N + damping * sum(scores[j] / float(len(graph.neighbors(j))) for j in linked_candidates)

            if abs(scores[candidate] - rank) <= convergence_threshold:
                convergences_achieved += 1

            scores[candidate] = rank

        if convergences_achieved == N:
            break

    return scores

def createGraph(n_grams, n_grammed_sentences):
    document_candidates = n_grams

    graph = nx.Graph()
    graph.add_nodes_from(document_candidates)

    # adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
    for sentence in n_grammed_sentences:
        for gram in sentence:
            for another_gram in sentence:
                if another_gram == gram:
                    continue
                else:
                    graph.add_edge(gram, another_gram)  # adding duplicate edges has no effect
    return graph

def getCandidatesfromDocumentSentences(n_grammed_sentences):
    document_candidates = []
    for sentence in n_grammed_sentences:
        for candidate in sentence:
            if candidate not in document_candidates:
                document_candidates.append(candidate)
    return document_candidates

def precision(list_r, list_a):
    i = set(list_r).intersection(list_a)
    len_a = len(list_a)
    if len_a == 0:
        return 0
    else:
        return float(len(i)) / float(len_a)

def recall(list_r, list_a):
    i = set(list_r).intersection(list_a)
    len_r = len(list_r)
    if len_r == 0:
        return 0
    else:
        return float(len(i)) / float(len_r)

def f1(prec, rec):
    if prec + rec == 0:
        return 0
    else:
        return float((2* rec * prec))/float((rec + prec))

def avg_precision(list_r, list_a):
    score = 0
    ri = 0
    for i in range(0, len(list_a)):
        if list_a[i] in list_r:
            ri = 1
        else:
            ri = 0
        score = score + (ri * precision(list_r, list_a[:i+1]))
    return float(score) / float(len(list_r))

def mean_avg_precision(ap_list):
    return float(sum(ap_list)) / float(len(ap_list))

# calculates the various evaluation values for a given document
def calculateDocumentEvaluation(docName, scores):
    retrieved = getTopCandidates(scores, 5)
    relevant = getDocumentRelevantKeyphrases(docName)
    values = {}
    values["precision"] = precision(relevant, retrieved)
    values["recall"] = recall(relevant, retrieved)
    values["f1"] = f1(values["precision"], values["recall"])
    values["ap"] = avg_precision(relevant, retrieved)
    return values

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

def calculateBM25Feature(docName, n_grammed_docs):
    FGngrams = n_grammed_docs[docName]
    BGngrams = n_grammed_docs.values()
    number_background_documents = len(BGngrams)
    average_document_length = retrieveAverageDocLength(BGngrams)
    inverted_index_dict = buildInvertedIndexDict(BGngrams)
    return performCandidateScoring(inverted_index_dict, FGngrams, average_document_length,
                            number_background_documents)

#######################################################################

def getKeyphrasesFromFile(filePathName):
    keyPhrases = readDocument(filePathName).splitlines()
    return keyPhrases

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

def getDocumentCandidates(docName, training_document=False):
    #returns a list of list of strings (each list contains the ngrams of a sentece)
    text = getDocumentContent(docName, training_document)
    sentences = PunktSentenceTokenizer().tokenize(text)
    result = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]
    return result

def getAllDocumentCandidates(docNames, training_documents=False):
    #returns a dictionary of docNames to lists of lists (docs - sentences - terms)
    allCandidates = {}
    for docName in docNames[:2]:
        print "\n>>Starting getDocumentCandidates('" + docName + "')" + strftime("%H:%M:%S", gmtime())
        allCandidates[docName] = getDocumentCandidates(docName, training_documents)
        print "##>>Ending getDocumentCandidates('" + docName + "')" + strftime("%H:%M:%S", gmtime())


    return allCandidates


# def auxiliarGetDocumentCandidates(args):
#     return (args[0], getDocumentCandidates(*args))
#
# def getAllDocumentCandidates(docNames, training_documents=False):
#     pool = ThreadPool(2)
#     job_args = [(docName, training_documents) for docName in docNames[:2]]
#     results = pool.map(auxiliarGetDocumentCandidates, job_args)
#     pool.close()
#     pool.join()
#     allCandidates = dict(results)
#
#     return allCandidates


def getWordVectors():
    word_vectors = {}

    with codecs.open(os.path.join(os.path.dirname(__file__), "resources", "glove.6B.50d.txt"), "r", "ISO-8859-1") as file:
        for line in file:
            try:
                idx = line.find(' ')
                str_vector = line[idx+1:].split(' ')
                word_vectors[line[:idx]] = map(float, str_vector)
            except:
                pass

    return word_vectors




