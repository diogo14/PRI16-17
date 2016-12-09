import networkx as nx
from scipy import spatial
from util import printTopCandidates
from util import calculateDocumentEvaluation
from util import mean_avg_precision
from util import calculateBM25Feature
from util import getCandidatesfromDocumentSentences
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import getWordVectors

from time import gmtime, strftime

from multiprocessing.dummy import Pool as ThreadPool

#code readability constants
SENTENCE_PRIOR_WEIGHTS = 1
BM25_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
SIMILARITY_EDGE_WEIGHTS = 4

memoized_similarity_weights = {}
print ">>Starting caching word vectors: " + strftime("%H:%M:%S", gmtime())
word_vector = getWordVectors()
print "##Ending caching word vectors: " + strftime("%H:%M:%S", gmtime())

def pagerank(graph, prior_weight_type, edge_weight_type):

    damping = 0.85
    N = graph.number_of_nodes()  # number of candidates
    convergence_threshold = 0.0001

    converged_candidates = set()

    scores = dict.fromkeys(graph.nodes(), 1.0 / N)  #initial value

    for _ in xrange(100):
        convergences_achieved = 0

        for candidate in graph.nodes():
            if candidate in converged_candidates:
                continue

            if prior_weight_type == SENTENCE_PRIOR_WEIGHTS:
                prior = graph.node[candidate]['sentence_prior_weight']
                prior_sum = graph.graph['sentence_prior_sum']
            else:
                prior = graph.node[candidate]['bm25_prior_weight']
                prior_sum = graph.graph['bm25_prior_sum']

            summatory = 0.0
            for j in graph.neighbors(candidate):
                if edge_weight_type == OCCURRENCE_EDGE_WEIGHTS:
                    weight = graph[j][candidate]['occurrence_weight']
                    weight_sum = graph.node[j]['occurrence_weight_sum']
                else:
                    weight = graph[j][candidate]['similarity_weight']
                    weight_sum = graph.node[j]['similarity_weight_sum']

                if weight != 0.0:
                    summatory += scores[j] * weight / weight_sum

            rank = (1 - damping) * prior / prior_sum - damping * summatory

            if abs(scores[candidate] - rank) <= convergence_threshold:
                convergences_achieved += 1
                converged_candidates.add(candidate)

            scores[candidate] = rank

        if convergences_achieved == N:
            break

    return scores

def computeSimilarityWeight(ngram1, ngram2):

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


def auxiliarCreateWeightedGraph(args):
    return (args[0], createWeightedGraph(*args))

def createWeightedGraph(docName, FGn_grammed_sentences, BGn_grammed_docs):
    print "\n>>Starting graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())

    print "\n>>Starting calculateBM25: " + strftime("%H:%M:%S", gmtime())
    scoresBM25 = calculateBM25Feature(docName, BGn_grammed_docs)
    document_candidates = BGn_grammed_docs[docName]
    print "##Ending calculateBM25: " + strftime("%H:%M:%S", gmtime())

    print "\n>>Starting adding nodes: " + strftime("%H:%M:%S", gmtime())
    graph = nx.Graph()
    for candidate in document_candidates:
        graph.add_node(candidate, bm25_prior_weight=scoresBM25[candidate])
    print "##Ending adding nodes: " + strftime("%H:%M:%S", gmtime())

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
                        graph.add_edge(gram, another_gram, occurrence_weight=1.0, similarity_weight=computeSimilarityWeight(gram, another_gram))
                        #print "##end adding edge '" + gram + " " + another_gram + ": " + strftime("%H:%M:%S", gmtime())
                    else:
                        graph[gram][another_gram]['occurrence_weight'] = graph[gram][another_gram][
                                                                             'occurrence_weight'] + 1.0  # additional occurrence of candidates
    print "\n>>Graph generated: " + strftime("%H:%M:%S", gmtime())

    print "\n>>Starting prior sum calc: " + strftime("%H:%M:%S", gmtime())
    graph.graph['sentence_prior_sum'] = sum(graph.node[k]['sentence_prior_weight'] for k in graph.nodes())
    graph.graph['bm25_prior_sum'] = sum(graph.node[k]['bm25_prior_weight'] for k in graph.nodes())
    print "##Ending prior sum calc: " + strftime("%H:%M:%S", gmtime())

    print "\n>>Starting weight sum calc: " + strftime("%H:%M:%S", gmtime())
    for candidate in graph.nodes():
        occurrence_sum = 0.0
        similarity_sum = 0.0
        for j in graph.neighbors(candidate):
            for k in graph.neighbors(j):
                occurrence_sum += graph[j][k]['occurrence_weight']
                similarity_sum += graph[j][k]['similarity_weight']

        graph.node[candidate]['occurrence_weight_sum'] = occurrence_sum
        graph.node[candidate]['similarity_weight_sum'] = similarity_sum

    print "##Ending weight sum calc: " + strftime("%H:%M:%S", gmtime())
    print "##Created graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())

    return graph
#######################################################################################################################

docNames = getDocumentNames()
print "\n>>Starting getAllDocumentCandidates: " + strftime("%H:%M:%S", gmtime())
candidates_in_sentences = getAllDocumentCandidates(docNames)
candidates = {}
print "\n"

for doc in candidates_in_sentences:
    candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])
    print "Candidates in '" + doc + "': " + str(len(candidates[doc]))

mean_avg_precs = {}

pool = ThreadPool(2)
job_args = [(docName, candidates_in_sentences[docName], candidates) for docName in docNames[:2]]
results = pool.map(auxiliarCreateWeightedGraph, job_args)
pool.close()
pool.join()

graphs = dict(results)

for docName in docNames[:2]:
    # print "\n>>Starting graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())
    # graph = createWeightedGraph(docName, candidates_in_sentences[docName], candidates)
    # print "##Created graph '" + docName + "' " + strftime("%H:%M:%S", gmtime())

    for variant_prior_weight in [SENTENCE_PRIOR_WEIGHTS, BM25_PRIOR_WEIGHTS]:
        for variant_edge_weight in [OCCURRENCE_EDGE_WEIGHTS, SIMILARITY_EDGE_WEIGHTS]:
            print ">>Starting pagerank '" + docName + "' " + strftime("%H:%M:%S", gmtime())
            scores = pagerank(graphs[docName], variant_prior_weight, variant_edge_weight)
            print "##Ending pagerank '" + docName + "' " + strftime("%H:%M:%S", gmtime())
            print ">>Starting calculateDocumentEvaluation '" + docName + "' " + strftime("%H:%M:%S", gmtime())
            values = calculateDocumentEvaluation(docName, scores)
            print "##Ending calculateDocumentEvaluation '" + docName + "' " + strftime("%H:%M:%S", gmtime())
            #print "====== " + docName + " ======"
            #print "Precision: " + str(values["precision"])
            #print "Recall: " + str(values["recall"])
            #print "F1: " + str(values["f1"]) + "\n"

            if variant_prior_weight not in mean_avg_precs:
                mean_avg_precs[variant_prior_weight] = {variant_edge_weight : [values['ap']]}
            elif variant_edge_weight not in mean_avg_precs[variant_prior_weight]:
                mean_avg_precs[variant_prior_weight][variant_edge_weight] = [values['ap']]
            else:
                mean_avg_precs[variant_prior_weight][variant_edge_weight].append(values['ap'])


for prior_variant, map_dict in mean_avg_precs.iteritems():
    for weight_variant, map_value in map_dict.iteritems():
        print "Mean Aver. Precision [" + str(prior_variant) + "][" + str(weight_variant) + "] : " + str(mean_avg_precision(map_value))

print "End: " + strftime("%H:%M:%S", gmtime())