import os
import networkx as nx
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import removePunctuation
from util import getWordGrams
from util import printTopCandidates


def calcPR(candidate, graph, candidate_scores):

    linked_candidates = graph.neighbors(candidate)   #set of candidates that co-occur with candidate
    number_linked_candidates = len(linked_candidates)    # |Links(Pj)|
    N = len(candidate_scores)            #number of candidates
    d = 0.5

    summatory = 0
    for neighbor_candidate in linked_candidates:
        summatory += candidate_scores[neighbor_candidate] / number_linked_candidates

    return d/N + (1-d) * summatory


#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = map(removePunctuation, PunktSentenceTokenizer().tokenize(document))   #with removed punctuation
n_grammed_sentences = [getWordGrams(sentence.split(' '), 1, 4) for sentence in sentences]

tokenized_document = nltk.word_tokenize(removePunctuation(document))
n_grammed_document = getWordGrams(tokenized_document, 1, 4)

g = nx.Graph()
g.add_nodes_from(n_grammed_document)

#adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
for sentence in n_grammed_sentences:
     for gram in sentence:
         for another_gram in sentence:
             if another_gram == gram:
                 continue
             else:
                 g.add_edge(gram, another_gram) #adding duplicate edges has no effect

#initializing each candidate score to 1
candidate_PR_scores = {}
for candidate in n_grammed_document:
    candidate_PR_scores[candidate] = 1

#iterative converging PR score calculation
for i in range(0, 50):
    for candidate in n_grammed_document:
        score = calcPR(candidate, g, candidate_PR_scores)
        candidate_PR_scores[candidate] = score


printTopCandidates(candidate_PR_scores, 5)
