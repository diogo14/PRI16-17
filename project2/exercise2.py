import networkx as nx
from scipy import spatial
from util import printTopCandidates
from util import calculateDocumentEvaluation
from util import mean_avg_precision
from util import createGraph
from util import getCandidatesfromDocumentSentences
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import getWordVectors

from time import gmtime, strftime

from multiprocessing.dummy import Pool as ThreadPool

#code readability constants
SIMPLE_WEIGHTS = 0
SENTENCE_PRIOR_WEIGHTS = 1
BM25_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
SIMILARITY_EDGE_WEIGHTS = 4

memoized_similarity_weights = {}
print ">>Starting caching word vectors: " + strftime("%H:%M:%S", gmtime())
word_vector = getWordVectors()
print "##Ending caching word vectors: " + strftime("%H:%M:%S", gmtime())



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
    return createGraph(docName, None, FGn_grammed_sentences, BGn_grammed_docs, True)




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