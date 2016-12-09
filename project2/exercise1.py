import os
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import getWordGrams
from util import printTopCandidates
from util import pagerank
from util import getCandidatesfromDocumentSentences
from util import createGraph

def createNotWeightedGraph(n_grams, n_grammed_sentences):
    return createGraph(None, n_grams, n_grammed_sentences, None, False)
#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = PunktSentenceTokenizer().tokenize(document)
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

candidate_scores = pagerank(createNotWeightedGraph(getCandidatesfromDocumentSentences(n_grammed_sentences), n_grammed_sentences),0,0)

printTopCandidates(candidate_scores, 10)

