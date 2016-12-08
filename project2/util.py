import nltk
import string
import codecs
from nltk.util import ngrams
from nltk.corpus import stopwords
import networkx as nx


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
                if len(word) == 1 or word.isnumeric() or word in stopwords.words('english') and not (len(ngram) == 3 and idx == 1):
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

def createGraph(n_grammed_sentences):
    document_candidates = getCandidatesfromDocumentSentences(n_grammed_sentences)

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
    total = 0
    for i in ap_list:
        total = total + i
    return float(total) / float(len(ap_list))