import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def readDocument(docPathName):
    file = open(docPathName, "r")
    text = file.read().lower()
    return text

#returns term dictionary whose values are indices of the IDF array also returned
#cleans stopwords, punctuation and uses bigrams
def calculate_idf(docs):
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2), smooth_idf=False, stop_words="english")
    vectorizer.fit_transform(docs)
    return vectorizer.vocabulary_, vectorizer.idf_

#returns term dictionary whose values are indices of the TF array also returned
#cleans stopwords, ponctuation and uses bigrams
def calculate_tf(doc):
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    tf_vec = vectorizer.fit_transform([doc])
    return vectorizer.vocabulary_, tf_vec

#calculates TFIDF scores for each foreground term
def calculate_scores(fore_dic, tf_vec, back_dic, idf_vec):
    scores = {}
    for term in fore_dic:
        score = tf_vec[(0, fore_dic[term])] * (idf_vec[back_dic[term]] - 1)
        scores[term] = score
    return scores

#receives the dictionary with terms and respective scores and returns a list with the top-5
def get_top_candidates(scores):
    list = []
    top = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    for tup in top[0:5]:
        list.append(tup)
    return list


def calculate_top_candidates(foreground_document, background_documents):
    # get background term list and IDF values
    back_dic, idf_vec = calculate_idf(background_documents)

    # get foreground term list and TF values
    fore_dic, tf_vec = calculate_tf(foreground_document)

    # calculate the score of each foreground term
    scores = calculate_scores(fore_dic, tf_vec, back_dic, idf_vec)

    return get_top_candidates(scores)
