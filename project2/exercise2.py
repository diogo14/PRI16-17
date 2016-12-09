import networkx as nx
from util import printTopCandidates
from util import calculateDocumentEvaluation
from util import mean_avg_precision
from util import createGraph
from util import getCandidatesfromDocumentSentences
from util import getDocumentNames
from util import getAllDocumentCandidates
from util import getWordVectors
from util import pagerank

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


def auxiliarCreateWeightedGraph(args):
    return (args[0], createWeightedGraph(*args))

def createWeightedGraph(docName, FGn_grammed_sentences, BGn_grammed_docs):
    return createGraph(docName, None, FGn_grammed_sentences, BGn_grammed_docs, True, memoized_similarity_weights, word_vector)


#######################################################################################################################

docNames = getDocumentNames()[:1]
print "\n>>Starting getAllDocumentCandidates: " + strftime("%H:%M:%S", gmtime())
candidates_in_sentences = getAllDocumentCandidates(docNames)
candidates = {}
print "\n"

for doc in candidates_in_sentences:
    candidates[doc] = getCandidatesfromDocumentSentences(candidates_in_sentences[doc])
    print "Candidates in '" + doc + "': " + str(len(candidates[doc]))

mean_avg_precs = {}

pool = ThreadPool(8)
job_args = [(docName, candidates_in_sentences[docName], candidates) for docName in docNames]
results = pool.map(auxiliarCreateWeightedGraph, job_args)
pool.close()
pool.join()

graphs = dict(results)

for docName in docNames:
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