import os
from sklearn.datasets import fetch_20newsgroups
from util import calculate_top_candidates
from util import readDocument

##################################################################
## Main starts here
##################################################################

#get foreground document
foreground_document = readDocument(os.path.join(os.path.dirname(__file__), "resources", "doc_ex1"))

#get training document set
training_dataset = fetch_20newsgroups(subset='train').data

#create background document set
background_documents = training_dataset + [foreground_document]

top_candidates = calculate_top_candidates(foreground_document, background_documents)

print "Term :::: TFIDF Score"
for tup in top_candidates:
        print str(tup[0]) + " :::: " + str(tup[1])


