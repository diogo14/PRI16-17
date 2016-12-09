import nltk
import codecs

import unicodedata
from nltk.util import ngrams
from nltk.corpus import stopwords
import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import math
import re
from scipy import spatial

from time import gmtime, strftime

#code readability constants
SIMPLE_WEIGHTS = 0
SENTENCE_PRIOR_WEIGHTS = 1
BM25_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
SIMILARITY_EDGE_WEIGHTS = 4

##########################PAGERANK########################

def pagerank(graph, prior_weight_type, edge_weight_type):

    damping = 0.85
    N = graph.number_of_nodes()  # number of candidates
    convergence_threshold = 0.0001

    converged_candidates = set()

    scores = dict.fromkeys(graph.nodes(), 1.0 / N)  #initial value

    for _ in xrange(50):
        convergences_achieved = 0

        for candidate in graph.nodes():
            if candidate in converged_candidates:
                continue

            if prior_weight_type == SENTENCE_PRIOR_WEIGHTS:
                prior = graph.node[candidate]['sentence_prior_weight']
                prior_sum = graph.graph['sentence_prior_sum']
            elif prior_weight_type == BM25_PRIOR_WEIGHTS:
                prior = graph.node[candidate]['bm25_prior_weight']
                prior_sum = graph.graph['bm25_prior_sum']
            else:
                prior = 1.0
                prior_sum = graph.graph['simple_prior_sum']

            summatory = 0.0
            for j in graph.neighbors(candidate):
                if edge_weight_type == OCCURRENCE_EDGE_WEIGHTS:
                    weight = graph[j][candidate]['occurrence_weight']
                    weight_sum = graph.node[j]['occurrence_weight_sum']
                elif edge_weight_type == SIMILARITY_EDGE_WEIGHTS:
                    weight = graph[j][candidate]['similarity_weight']
                    weight_sum = graph.node[j]['similarity_weight_sum']
                else:
                    weight = 1.0
                    weight_sum = graph.node[j]['simple_weight_sum']

                if weight != 0.0:
                    summatory += scores[j] * weight / weight_sum

            rank = (1 - damping) * prior / prior_sum + damping * summatory

            if abs(scores[candidate] - rank) <= convergence_threshold:
                convergences_achieved += 1
                converged_candidates.add(candidate)

            scores[candidate] = rank

        if convergences_achieved == N:
            break

    return scores


def computeSimilarityWeight(ngram1, ngram2, memoized_similarity_weights, word_vector):

    zero_vector = [0.0 for _ in range(50)]

    if ngram1 in memoized_similarity_weights and ngram2 in memoized_similarity_weights[ngram1]:
        return memoized_similarity_weights[ngram1][ngram2]
    elif ngram2 in memoized_similarity_weights and ngram1 in memoized_similarity_weights[ngram2]:
        return memoized_similarity_weights[ngram2][ngram1]

    for word in ngram1.split(' '):
        ngram_vectors = []
        if word in word_vector:
            ngram_vectors.append(word_vector[word])
        else:
            ngram_vectors.append(zero_vector)
        avg_ngram1_vector = map(lambda x: sum(x) / float(len(x)), zip(*ngram_vectors))

    for word in ngram2.split(' '):
        ngram_vectors = []
        if word in word_vector:
            ngram_vectors.append(word_vector[word])
        else:
            ngram_vectors.append(zero_vector)
        avg_ngram2_vector = map(lambda x: sum(x) / float(len(x)), zip(*ngram_vectors))

    if sum(avg_ngram1_vector) == 0.0 or sum(avg_ngram2_vector) == 0.0:  #both non-zeroed vectors
        similarity_weight = 0.0
    else:
        similarity_weight = 1 - spatial.distance.cosine(avg_ngram1_vector, avg_ngram2_vector) + 1

    if ngram1 not in memoized_similarity_weights:
        memoized_similarity_weights[ngram1] = {ngram2:similarity_weight}
    else:
        memoized_similarity_weights[ngram1][ngram2] = similarity_weight

    return similarity_weight

def createNotWeightedGraph(n_grams, n_grammed_sentences):
    return createGraph(None, n_grams, n_grammed_sentences, None, False, None, None)

def createGraph(docName, FGn_grams, FGn_grammed_sentences, BGn_grammed_docs, weighted, memoized_similarity_weights, word_vector):
    #print "\n>>Starting graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())
    if(weighted):
        print "\n>>Starting calculateBM25: " + strftime("%H:%M:%S", gmtime())
        scoresBM25 = calculateBM25Feature(docName, BGn_grammed_docs)
        document_candidates = BGn_grammed_docs[docName]
        print "##Ending calculateBM25: " + strftime("%H:%M:%S", gmtime())

        print "\n>>Starting adding nodes: " + strftime("%H:%M:%S", gmtime())
        graph = nx.Graph()
        for candidate in document_candidates:
            graph.add_node(candidate, bm25_prior_weight=scoresBM25[candidate])
        print "##Ending adding nodes: " + strftime("%H:%M:%S", gmtime())
        graph.graph['simple_prior_sum'] = len(document_candidates)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(FGn_grams)
        graph.graph['simple_prior_sum'] = len(FGn_grams)

    if(weighted):
        # adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
        # adding prior weight for each candidate (based on the sentence it appears by the first time
        for idx, sentence in enumerate(FGn_grammed_sentences):
            prior_weight = len(FGn_grammed_sentences) - idx  # first sentences have better candidates
            for gram in sentence:
                if 'sentence_prior_weight' not in graph[gram]:  # set only by the first time
                    graph.node[gram]['sentence_prior_weight'] = float(prior_weight)
                for another_gram in sentence:
                    if another_gram == gram:
                        continue
                    else:
                        if not graph.has_edge(gram, another_gram):
                            #print ">>start adding edge '" + gram + " " + another_gram + ": " + strftime("%H:%M:%S", gmtime())
                            graph.add_edge(gram, another_gram, occurrence_weight=1.0,
                                    similarity_weight=computeSimilarityWeight(gram, another_gram, memoized_similarity_weights, word_vector))
                            #print "##end adding edge '" + gram + " " + another_gram + ": " + strftime("%H:%M:%S", gmtime())
                        else:
                            graph[gram][another_gram]['occurrence_weight'] = graph[gram][another_gram][
                                                                             'occurrence_weight'] + 1.0  # additional occurrence of candidates

    else:
        # adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
        for sentence in FGn_grammed_sentences:
            for gram in sentence:
                for another_gram in sentence:
                    if another_gram == gram:
                        continue
                    else:
                        graph.add_edge(gram, another_gram)  # adding duplicate edges has no effect


    print "\n>>Graph generated: " + strftime("%H:%M:%S", gmtime())
    if(weighted):
        print "\n>>Starting prior sum calc: " + strftime("%H:%M:%S", gmtime())
        graph.graph['sentence_prior_sum'] = sum(graph.node[k]['sentence_prior_weight'] for k in graph.nodes())
        graph.graph['bm25_prior_sum'] = sum(graph.node[k]['bm25_prior_weight'] for k in graph.nodes())
        print "##Ending prior sum calc: " + strftime("%H:%M:%S", gmtime())

        print "\n>>Starting weight sum calc: " + strftime("%H:%M:%S", gmtime())
        for candidate in graph.nodes():
            graph.node[candidate]['simple_weight_sum'] = 0
            occurrence_sum = 0.0
            similarity_sum = 0.0
            for j in graph.neighbors(candidate):
                graph.node[candidate]['simple_weight_sum'] += 1
                occurrence_sum += graph[j][candidate]['occurrence_weight']
                similarity_sum += graph[j][candidate]['similarity_weight']

            graph.node[candidate]['occurrence_weight_sum'] = occurrence_sum
            graph.node[candidate]['similarity_weight_sum'] = similarity_sum
    else:
        for candidate in graph.nodes():
            graph.node[candidate]['simple_weight_sum'] = len(graph.neighbors(candidate))

    print "##Ending weight sum calc: " + strftime("%H:%M:%S", gmtime())
    #print "##Created graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())

    return graph

######################Evaluation#################################

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
    retrieved = [t[0] for t in retrieved]
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
    for docName in docNames:
        print "\n>>Starting getDocumentCandidates('" + docName + "')" + strftime("%H:%M:%S", gmtime())
        allCandidates[docName] = getDocumentCandidates(docName, training_documents)
        print "##>>Ending getDocumentCandidates('" + docName + "')" + strftime("%H:%M:%S", gmtime())


    return allCandidates

def getCandidatesfromDocumentSentences(n_grammed_sentences):
    document_candidates = []
    for sentence in n_grammed_sentences:
        for candidate in sentence:
            if candidate not in document_candidates:
                document_candidates.append(candidate)
    return document_candidates

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
                ignored = ['P', 'S', 'Z', 'C']
                invalid_begin_end = unicodedata.category(word[0])[0] in ignored or \
                                  unicodedata.category(word[len(word)-1])[0] in ignored

                if invalid_begin_end or not has_letter or word in stopwords.words('english') and not (len(ngram) == 3 and idx == 1):
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
        print("" + candidate[0].encode("ISO-8859-1") + " - " + str(candidate[1]))


def getOrderedCandidates(scores):   #decreasing order
    # reverse ordering of candidates scores
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)



