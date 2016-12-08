import os
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

from util import readDocument
from util import removePunctuation
from util import getWordGrams
from util import printTopCandidates
from util import createGraph
from util import pagerank

#######################################################################################################################

document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

sentences = PunktSentenceTokenizer().tokenize(document)   #with removed punctuation
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

candidate_scores = pagerank(createGraph(n_grammed_sentences))

printTopCandidates(candidate_scores, 10)

