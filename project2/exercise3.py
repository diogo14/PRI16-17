from numpy import dot
import os
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from util import readDocument
from util import removePunctuation
from util import getWordGrams

def getKeyphrasesFromFile(filePathName):
    keyPhrases = readDocument(filePathName).splitlines()
    return keyPhrases

def getDocumentNames(training_document=False):
    if (training_document == False):
        path = os.path.join(os.path.dirname(__file__), "dataset", "documents")
    else:
        path = os.path.join(os.path.dirname(__file__), "training", "documents")
    fileNames = os.listdir(path)
    fileNames.sort()
    return fileNames

def getDocumentContent(docName, training_document=False):
    if (training_document == False):
        path = os.path.join(os.path.dirname(__file__), "dataset", "documents", docName)
    else:
        path = os.path.join(os.path.dirname(__file__), "training", "documents", docName)
    return readDocument(path)

def getDocumentRelevantKeyphrases(docName, training_document=False):
    if(training_document == False):
        rootPath = os.path.join(os.path.dirname(__file__), "dataset", "indexers")
        k = []
        for i in range(1, 7):
            k.append(getKeyphrasesFromFile(os.path.join(rootPath, "iic" + str(i), docName[:-3] + "key")))
        return list(set().union(k[0], k[1], k[2], k[3], k[4], k[5]))
    else:
        rootPath = os.path.join(os.path.dirname(__file__), "training", "keys")
        return getKeyphrasesFromFile(os.path.join(rootPath, docName[:-3] + "key"))

def getDocumentCandidates(docName, training_document=False):
    #returns a list of list of strings (each list contains the ngrams of a sentece)
    text = getDocumentContent(docName, training_document)
    sentences = map(removePunctuation, PunktSentenceTokenizer().tokenize(text))  # with removed punctuation
    return [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

def getAllDocumentCandidates(docNames, training_documents=False):
    allCandidates = {}
    for docName in docNames:
        allCandidates[docName] = getDocumentCandidates(docName, training_documents)
    return allCandidates

def generateTrainingData(trainingDocuments):


def PRank(training_data):
    # receives a list of tuples. Each tuple is composed of a list of floats and a boolean (list, boolean)
    # the list of floats contains the candidates' features and the boolean is True if the candidate is a keyword

    # returns the weight vector and the threshold

    w = [0] * len(training_data[0][0]) #weight vector
    b = 0 #threshold

    for x in training_data:
        if (dot(w, x[0]) < b):
            yp = False
        else:
            yp = True

        y = x[1]

        if(yp != y):

            if(y==False):
                yr = -1
            else:
                yr = 1

            if(yr*(dot(w,x[0])-b) <= 0):
                for i in range(0, len(w)):
                    w[i] = w[i] + yr * x[0][i]
                b -= yr

    return w, b
