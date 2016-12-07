import os
import networkx as nx
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import removePunctuation
from util import getWordGrams
from util import printTopCandidates


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

#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = map(removePunctuation, PunktSentenceTokenizer().tokenize(document))   #with removed punctuation
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

document_candidates = []
for sentence in n_grammed_sentences:
    for candidate in sentence:
        if candidate not in document_candidates:
            document_candidates.append(candidate)


graph = nx.Graph()
graph.add_nodes_from(document_candidates)

#adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
for sentence in n_grammed_sentences:
     for gram in sentence:
         for another_gram in sentence:
             if another_gram == gram:
                 continue
             else:
                 graph.add_edge(gram, another_gram) #adding duplicate edges has no effect


candidate_scores = pagerank(graph)

printTopCandidates(candidate_scores, 10)

