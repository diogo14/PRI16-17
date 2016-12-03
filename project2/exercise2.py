import os
import networkx as nx
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import removePunctuation
from util import getWordGrams
from util import printTopCandidates

#code readability constants
SENTENCE_PRIOR_WEIGHTS = 1
TFIDF_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
DISTRIBUTIONAL_EDGE_WEIGHTS = 4


def calcPR(candidate, graph, candidate_scores, prior_weight_type, edge_weight_type):

    linked_candidates = graph.neighbors(candidate)   #set of candidates that co-occur with candidate
    d = 0.5

    if prior_weight_type == SENTENCE_PRIOR_WEIGHTS:
        denominator = 0
        for neighbor_candidate in linked_candidates:
            denominator += candidate_scores[neighbor_candidate]['sentence_prior_weight']

        prior_weight_part = candidate_scores[candidate]['sentence_prior_weight'] / denominator
    else:
        prior_weight_part = 0 #TODO TFIDF_PRIOR_WEIGHTS


    if edge_weight_type == OCCURRENCE_EDGE_WEIGHTS:
        edge_weight_part = 0
        for neighbor_candidate in linked_candidates:
            numerator = candidate_scores[neighbor_candidate] * graph[candidate][neighbor_candidate]['weight']

            denominator = 0
            for linked_to_neighbor_candidate in graph.neighbors(neighbor_candidate):
                    denominator += graph[neighbor_candidate][linked_to_neighbor_candidate]['weight']

            edge_weight_part += numerator / denominator
    else:
        edge_weight_part = 0 #TODO DISTRIBUTIONAL_EDGE_WEIGHTS



    return d * prior_weight_type + (1 - d) * edge_weight_part



#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = map(removePunctuation, PunktSentenceTokenizer().tokenize(document))   #with removed punctuation
n_grammed_sentences = [getWordGrams(sentence.split(' '), 1, 4) for sentence in sentences]

tokenized_document = nltk.word_tokenize(removePunctuation(document))
n_grammed_document = getWordGrams(tokenized_document, 1, 4)

g = nx.Graph()
g.add_nodes_from(n_grammed_document)


#initializing each candidate score to 1
candidate_scores = {}
for candidate in n_grammed_document:
    candidate_scores[candidate] = {'PR' : 1}

#adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
#adding prior weight for each candidate (based on the sentence it appears by the first time
for idx, sentence in enumerate(n_grammed_sentences):
     sentence_prior_weight = len(n_grammed_document) - idx   #first sentences have better candidates

     for gram in sentence:

         if 'sentence_prior_weight' not in candidate_scores[gram]:  #only adding prior weight by the first time (greater sentence score)
            candidate_scores[gram]['sentence_prior_weight'] = sentence_prior_weight

         for another_gram in sentence:
             if another_gram == gram:
                 continue
             else:
                if not g.has_edge(gram, another_gram):
                    g.add_edge(gram, another_gram, weight=1)
                else:
                    g[gram][another_gram]['weight'] = g[gram][another_gram]['weight'] + 1   #additional occurrence


#TO BE CONTINUED...