import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def readDocument(docName, directory=os.path.join(os.path.dirname(__file__), os.pardir, "resources\\")):
    """
    Reads a given document from 'resources' directory by default
    """

    file = open(directory + docName, "r")
    text = file.read().lower()
    no_punctuation = text #TODO remove punctuation

    return no_punctuation

#training data
train = fetch_20newsgroups(subset='train')

#document to apply automatic keyphrase extraction on
testDoc = readDocument("doc1")

#matrix containing document in each row and term in each column vectorizer[doc][term] = tf-idf-value
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", use_idf=True )
train_vec = vectorizer.fit_transform(train.data[:2])    #TODO remove :2 to include all data
test_vec = vectorizer.transform([testDoc])

#collecting candidate rankings
feature_names = vectorizer.get_feature_names()
scored_candidates = {}
for idx, candidate in enumerate(feature_names):
    scored_candidates[candidate] = test_vec.toarray()[0][idx] #adding all candidates with respective tf-idf value to the dictionary

#reverse ordering of candidates
top_candidates = sorted(scored_candidates.items(), key = lambda x: x[1], reverse = True)

#top 5 candidates
for candidate in top_candidates[:5]:
    print("" + str(candidate[0]) + " - " + str(candidate[1]))


