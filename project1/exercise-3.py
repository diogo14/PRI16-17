import os
import re
import nltk
from nltk.util import ngrams

DOCUMENTS_DIRECTORY = os.path.join(os.path.dirname(__file__), os.pardir, "resources\\")
def readDocument(docName):
    file = open(DOCUMENTS_DIRECTORY + docName, "r")
    text = file.read().lower()
    no_punctuation = text #TODO remove punctuation
    return no_punctuation

#getting n-grams in specified range
def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

#filtering each n-gram candidate by a given regex
def filter_n_grams(list, regex_pattern):
    filtered = []

    for candidate in list:
        tagged = nltk.pos_tag(nltk.word_tokenize(candidate))
        observed_pattern = ""
        for pair in tagged:
            observed_pattern += " " + pair[1]
        print(tagged)
        print(observed_pattern)

        if(re.match(regex_pattern, observed_pattern)):
            filtered.append(candidate)

    return filtered


doc = readDocument("doc1")
n_grams = word_grams(nltk.word_tokenize(doc))

filtering_regex = r""   #TODO choose good pattern to match
filtered_n_grams = filter_n_grams(n_grams, filtering_regex)

print(filtered_n_grams)


#to be continued...