from util import calculateDocumentEvaluation
from util import mean_avg_precision
from util import createGraph
from util import getCandidatesfromDocumentSentences
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import getWordVectors
from util import pagerank

from multiprocessing.dummy import Pool as ThreadPool

#code readability constants
SENTENCE_PRIOR_WEIGHTS = 1
BM25_PRIOR_WEIGHTS = 2
OCCURRENCE_EDGE_WEIGHTS = 3
SIMILARITY_EDGE_WEIGHTS = 4

#cached results
memoized_similarity_weights = {}
word_vector = getWordVectors()

def auxiliarCreateWeightedGraph(args):
    return (args[0], createWeightedGraph(*args))

def createWeightedGraph(docName, FGn_grammed_sentences, BGn_grammed_docs):
    return createGraph(docName, None, FGn_grammed_sentences, BGn_grammed_docs, True, memoized_similarity_weights, word_vector)

#######################################################################################################################

docNames = getDocumentNames()
candidates_in_sentences = getAllDocumentCandidates(docNames)
candidates = {}

for doc in candidates_in_sentences:
    candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])

mean_avg_precs = {}

pool = ThreadPool(8)
job_args = [(docName, candidates_in_sentences[docName], candidates) for docName in docNames]
results = pool.map(auxiliarCreateWeightedGraph, job_args)
pool.close()
pool.join()

graphs = dict(results)

for docName in docNames:
    for variant_prior_weight in [SENTENCE_PRIOR_WEIGHTS, BM25_PRIOR_WEIGHTS]:
        for variant_edge_weight in [OCCURRENCE_EDGE_WEIGHTS, SIMILARITY_EDGE_WEIGHTS]:
            scores = pagerank(graphs[docName], variant_prior_weight, variant_edge_weight)
            values = calculateDocumentEvaluation(docName, scores)

            if variant_prior_weight not in mean_avg_precs:
                mean_avg_precs[variant_prior_weight] = {variant_edge_weight : [values['ap']]}
            elif variant_edge_weight not in mean_avg_precs[variant_prior_weight]:
                mean_avg_precs[variant_prior_weight][variant_edge_weight] = [values['ap']]
            else:
                mean_avg_precs[variant_prior_weight][variant_edge_weight].append(values['ap'])


for prior_variant, map_dict in mean_avg_precs.iteritems():
    for weight_variant, map_value in map_dict.iteritems():
        print "Mean Aver. Precision [" + str(prior_variant) + "][" + str(weight_variant) + "] : " + str(mean_avg_precision(map_value))
