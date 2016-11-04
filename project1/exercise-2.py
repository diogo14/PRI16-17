import nltk
import numpy
import os
from util import readDocument
from util import calculate_top_candidates

# returns a list of the keyphrases listed in the file specified
def getKeyphrasesFromFile(filePathName):
    keyPhrases = readDocument(filePathName).splitlines()
    return keyPhrases

# returns a list of all the keyphrases relevant to a given document
def getDocumentRelevantKeyphrases(docName):
    rootPath = os.path.join(os.path.dirname(__file__), "dataset", "indexers")
    k = []
    for i in range(1, 7):
        k.append(getKeyphrasesFromFile(os.path.join(rootPath, "iic" + str(i), docName[:-3] + "key")))
    return list(set().union(k[0], k[1], k[2], k[3], k[4], k[5]))

# calculates the various evaluation values for a given document
def calculateDocumentEvaluation(foreground_document, background_documents):
    retrieved =  [x[0] for x in calculate_top_candidates(foreground_document["text"], background_documents)]
    relevant = getDocumentRelevantKeyphrases(foreground_document["name"])
    values = {}
    values["precision"] = precision(relevant, retrieved)
    values["recall"] = recall(relevant, retrieved)
    values["f1"] = f1(values["precision"], values["recall"])
    values["ap"] = avg_precision(relevant, retrieved)
    return values

# returns a list of all the file names of a given directory
def getDocumentNames(path=os.path.join(os.path.dirname(__file__), "dataset", "documents")):
    fileNames = os.listdir(path)
    fileNames.sort()
    return fileNames

# calculates and prints the evaluation results for all the docs in the collection
def print_evaluations(documentNames):
    rootPath = os.path.join(os.path.dirname(__file__), "dataset", "documents")
    documents = {}
    background_documents = []
    for docName in documentNames:
        text = readDocument(os.path.join(rootPath, docName))
        utf = unicode(text, "ISO-8859-1")
        documents[docName] = utf
        background_documents.append(utf)

    foreground_document = {}
    map = []
    for docName in documents:
        foreground_document["name"] = docName
        foreground_document["text"] = documents[docName]
        values = calculateDocumentEvaluation(foreground_document, background_documents)
        print "====== " + docName + " ======"
        print "Precision: " + str(values["precision"])
        print "Recall: " + str(values["recall"])
        print "F1: " + str(values["f1"]) + "\n"
        map.append(values["ap"])

    print "==================="
    print "Mean Av. Precision: " + str(mean_avg_precision(map))

def precision(list_r, list_a):
    i = set(list_r).intersection(list_a)
    len_a = len(list_a)
    if len_a == 0:
        return 0
    else:
        return float(len(i)) / float(len_a)

def recall(list_r, list_a):
    i = set(list_r).intersection(list_a)
    len_r = len(list_r)
    if len_r == 0:
        return 0
    else:
        return float(len(i)) / float(len_r)

def f1(prec, rec):
    if prec + rec == 0:
        return 0
    else:
        return float((2* rec * prec))/float((rec + prec))

def avg_precision(list1, list2, k=5):

    if len(list2) > k:
        list2 = list2[:k]

    score = 0.0
    hits = 0.0

    for i, p in enumerate(list2):
        if p in list1 and p not in list2[:i]:
            hits += 1.0
            score += hits / (i + 1.0)

    if not list1:
        return 0.0

    return float(score) / min(len(list1), k)

def mean_avg_precision(ap_list):
    total = 0
    for i in ap_list:
        total = total + i
    return float(total) / float(len(ap_list))

##################################################################
## Main starts here
##################################################################
#print_evaluations(getDocumentNames())

print recall(["A", "B", "C", "D"], ["A", "E", "F", "B", "Z"])
