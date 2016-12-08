import os
import networkx as nx
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import getWordGrams
from util import printTopCandidates

#code readability constants
SENTENCE_PRIOR_WEIGHTS = 1
TFIDF_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
DISTRIBUTIONAL_EDGE_WEIGHTS = 4


def pagerank(graph, prior_weight_type, edge_weight_type):

    damping = 0.85
    N = graph.number_of_nodes()  # number of candidates
    convergence_threshold = 0.0001

    scores = dict.fromkeys(graph.nodes(), 1.0 / N)  #initial value

    for _ in xrange(100):
        convergences_achieved = 0

        for candidate in graph.nodes():
            if prior_weight_type == SENTENCE_PRIOR_WEIGHTS:
                prior = graph.node[candidate]['sentence_prior_weight']
                prior_sum = sum(graph.node[k]['sentence_prior_weight'] for k in graph.nodes())
            else:
                prior = 0   #TODO TFIDF_PRIOR_WEIGHTS
                prior_sum = 0.0

            summatory = 0.0
            for j in graph.neighbors(candidate):
                if edge_weight_type == OCCURRENCE_EDGE_WEIGHTS:
                    weight = graph[j][candidate]['occurrence_weight']
                    weight_sum = sum(graph[j][k]['occurrence_weight'] for k in graph.neighbors(j))
                else:
                    # TODO DISTRIBUTIONAL
                    weight = 0.0
                    weight_sum = 0.0

                summatory += scores[j] * weight / weight_sum

            rank = (1 - damping) * prior / prior_sum - damping * summatory

            if abs(scores[candidate] - rank) <= convergence_threshold:
                convergences_achieved += 1

            scores[candidate] = rank

        if convergences_achieved == N:
            break

    return scores




#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = PunktSentenceTokenizer().tokenize(document)
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

document_candidates = []
for sentence in n_grammed_sentences:
    for candidate in sentence:
        if candidate not in document_candidates:
            document_candidates.append(candidate)


graph = nx.Graph()
graph.add_nodes_from(document_candidates)


#adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
#adding prior weight for each candidate (based on the sentence it appears by the first time
for idx, sentence in enumerate(n_grammed_sentences):
    prior_weight = len(n_grammed_sentences) - idx        #first sentences have better candidates
    for gram in sentence:
        if 'sentence_prior_weight' not in graph[gram]:   #set only by the first time
            graph.node[gram]['sentence_prior_weight'] = float(prior_weight)

        for another_gram in sentence:
            if another_gram == gram:
                continue
            else:
                if not graph.has_edge(gram, another_gram):
                    graph.add_edge(gram, another_gram, occurrence_weight=1.0)
                else:
                    graph[gram][another_gram]['occurrence_weight'] = graph[gram][another_gram]['occurrence_weight'] + 1.0   #additional occurrence of candidates


candidate_scores = pagerank(graph, SENTENCE_PRIOR_WEIGHTS, OCCURRENCE_EDGE_WEIGHTS)
printTopCandidates(candidate_scores, 10)

