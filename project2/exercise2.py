import networkx as nx
from scipy import spatial
from util import printTopCandidates
from util import calculateDocumentEvaluation
from util import mean_avg_precision
from util import calculateBM25Feature
from util import getCandidatesfromDocumentSentences
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import getWordVector


#code readability constants
SENTENCE_PRIOR_WEIGHTS = 1
BM25_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
SIMILARITY_EDGE_WEIGHTS = 4


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
                prior = graph.node[candidate]['bm25_prior_weight']
                prior_sum = sum(graph.node[k]['bm25_prior_weight'] for k in graph.nodes())

            summatory = 0.0
            for j in graph.neighbors(candidate):
                if edge_weight_type == OCCURRENCE_EDGE_WEIGHTS:
                    weight = graph[j][candidate]['occurrence_weight']
                    weight_sum = sum(graph[j][k]['occurrence_weight'] for k in graph.neighbors(j))
                else:
                    weight = graph[j][candidate]['similarity_weight']
                    weight_sum = sum(graph[j][k]['similarity_weight'] for k in graph.neighbors(j))

                summatory += scores[j] * weight / weight_sum

            rank = (1 - damping) * prior / prior_sum - damping * summatory

            if abs(scores[candidate] - rank) <= convergence_threshold:
                convergences_achieved += 1

            scores[candidate] = rank

        if convergences_achieved == N:
            break

    return scores

def computeSimilarityWeight(word1, word2):

    vector1 = getWordVector(word1)
    vector2 = getWordVector(word2)

    if vector1 is None or vector2 is None:
        return 0.0
    else:
        return 1 - spatial.distance.cosine(vector1, vector2) + 1


def createWeightedGraph(docName, FGn_grammed_sentences, BGn_grammed_docs):

    scoresBM25 = calculateBM25Feature(docName, BGn_grammed_docs)

    document_candidates = BGn_grammed_docs[docName]

    graph = nx.Graph()
    for candidate in document_candidates:
        graph.add_node(candidate, bm25_prior_weight=scoresBM25[candidate])

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
                        graph.add_edge(gram, another_gram, occurrence_weight=1.0, similarity_weight=computeSimilarityWeight(gram, another_gram))
                    else:
                        graph[gram][another_gram]['occurrence_weight'] = graph[gram][another_gram][
                                                                             'occurrence_weight'] + 1.0  # additional occurrence of candidates
    return graph
#######################################################################################################################


docNames = getDocumentNames()
candidates_in_sentences = getAllDocumentCandidates(docNames)
candidates = {}
for doc in candidates_in_sentences:
    candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])

for variant_prior_weight in [SENTENCE_PRIOR_WEIGHTS, BM25_PRIOR_WEIGHTS]:
    for variant_edge_weight in [OCCURRENCE_EDGE_WEIGHTS, SIMILARITY_EDGE_WEIGHTS]:
        for docName in docNames:
            graph = createWeightedGraph(docName, candidates_in_sentences[docName], candidates)
            scores = pagerank(graph, variant_prior_weight, variant_edge_weight)
            values = calculateDocumentEvaluation(docName, scores)
            #print "====== " + docName + " ======"
            #print "Precision: " + str(values["precision"])
            #print "Recall: " + str(values["recall"])
            #print "F1: " + str(values["f1"]) + "\n"
            map.append(values["ap"])
        if (variant_prior_weight == SENTENCE_PRIOR_WEIGHTS and variant_edge_weight == OCCURRENCE_EDGE_WEIGHTS):
            print "SENTENCE_PRIOR_WEIGHTS & OCCURRENCE_EDGE_WEIGHTS"
        elif (variant_prior_weight == SENTENCE_PRIOR_WEIGHTS and variant_edge_weight == SIMILARITY_EDGE_WEIGHTS):
            print "SENTENCE_PRIOR_WEIGHTS & SIMILARITY_EDGE_WEIGHTS"
        elif (variant_prior_weight == BM25_PRIOR_WEIGHTS and variant_edge_weight == OCCURRENCE_EDGE_WEIGHTS):
            print "BM25_PRIOR_WEIGHTS & OCCURRENCE_EDGE_WEIGHTS"
        else:
            print "BM25_PRIOR_WEIGHTS & SIMILARITY_EDGE_WEIGHTS"
        print "Mean Av. Precision: " + str(mean_avg_precision(map))
        print "==================="