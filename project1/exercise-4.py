import re
import os
import math
import operator

from nltk.corpus import stopwords

def readDocument(docName, directory=os.path.join(os.path.dirname(__file__), os.pardir, "resources\\")):
    """
    Reads a given document from 'resources' directory by default
    """

    file = open(directory + docName, "r")
    text = file.read().lower()
    no_punctuation = text #TODO remove punctuation

    return no_punctuation

def tokenize(t):

    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãå&@#A-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds

def getAllNgrams(text,maxn) :

    words = tokenize(text)
    i=0
    terms = dict()

    for word in words :
        if word not in stopwords.words('english') and len(word) > 1 and '@' not in word:
            if word in terms :
                terms[word] += 1
            else :
                terms[word] = 1
        if maxn >= 2 :
            if i< len(words)-1 :
                if words[i] not in stopwords.words('english') and words[i+1] not in stopwords.words('english'):
                    bigram = words[i]+ " " +words[i+1]
                    if bigram in terms :
                        terms[bigram] += 1
                    else :
                        terms[bigram] = 1

                if maxn >= 3 :
                    if i < len(words)-2 :
                        if not words[i] in stopwords.words('english') and not words[i+2] in stopwords.words('english'):
                            # middle word can be a stopword
                            trigram = words[i]+ " " +words[i+1]+ " " +words[i+2]
                            if trigram in terms :
                                terms[trigram] += 1
                            else :
                                terms[trigram] = 1
        i += 1
    return terms

def read_text_in_dict(text):

    freq_dict = getAllNgrams(text,3)
    total_term_count = 0

    for key in freq_dict:
        total_term_count += freq_dict[key]

    return freq_dict, total_term_count

def print_top_n_terms(score_dict,n):

    sorted_terms = sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)
    i=0

    for (term,score) in sorted_terms:
        i += 1
        print(term,score)
        if i==n:
            break


##################################################################
## Main starts here
##################################################################

#building structure (dict) to hold each term (n-gram) ocurrance number
#for both foreground and background corpus
fg_dict, fg_term_count = read_text_in_dict(readDocument("doc_ex4"))
bg_dict, bg_term_count = read_text_in_dict("backgound text") #TODO background collection

#calculate kldiv per term (n-gram) from foreground document
kldiv_per_term = {}

for term in fg_dict:
    fg_freq = fg_dict[term]

    # kldivI is kldiv for informativeness: relative to bg corpus freqs
    bg_freq = 1

    if term in bg_dict:
        bg_freq = bg_dict[term]

    relfreq_fg = float(fg_freq)/float(fg_term_count)
    relfreq_bg = float(bg_freq)/float(bg_term_count)

    kldivI = relfreq_fg*math.log(relfreq_fg/relfreq_bg)

    # kldivP is kldiv for phraseness: relative to unigram freqs
    unigrams = term.split(" ")
    relfreq_unigrams = 1.0

    for unigram in unigrams:
        if unigram in fg_dict:  # testing because middle n-gram word can be a stop word, so it has no occurance in fg_dict
            u_freq = fg_dict[unigram]
            u_relfreq = float(u_freq)/float(fg_term_count)
            relfreq_unigrams *= u_relfreq

    kldivP = relfreq_fg*math.log(relfreq_fg/relfreq_unigrams)
    kldiv = kldivI+kldivP

    kldiv_per_term[term] = kldiv

    #print(term,kldivI,kldivP,kldiv)