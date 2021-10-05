#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:11:54 2021

@author: yannis.coutouly
"""

"""
This file is for looking the compatibility in the W2VF embeddings

"""

import heapq
import numpy as np
import tensorflow as tf

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

class Embeddings:
   def __init__(self, vecsfile, vocabfile=None, normalize=True):
      if vocabfile is None: vocabfile = vecsfile.replace("npy","vocab")
      self._vecs = np.load(vecsfile)
      self._vocab = open(vocabfile).read().split()
      if normalize:
         self._vecs = ugly_normalize(self._vecs)
      self._w2v = {w:i for i,w in enumerate(self._vocab)}

   @classmethod
   def load(cls, vecsfile, vocabfile=None):
      return Embeddings(vecsfile, vocabfile)

   def word2vec(self, w):
      return self._vecs[self._w2v[w]]

   def similar_to_vec(self, v, N=10):
      sims = self._vecs.dot(v)
      sims = heapq.nlargest(N, zip(sims,self._vocab,self._vecs))
      return sims

   def most_similar(self, word, N=10):
      w = self._vocab.index(word)
      sims = self._vecs.dot(self._vecs[w])
      sims = heapq.nlargest(N, zip(sims,self._vocab))
      return sims

###############################################################################

generalDataPath = "../BatchExp/SecondBatchExp/DataExperience/"
fileToTestPath = generalDataPath + "Test/test_N1=8_Ot=8"
pathToExp  = "../BatchExp/SecondBatchExp/Experience/Compos_W2VF_test_N1=8but10N1=5/"


fixedNouns = ["chiot", "agriculteur", "étoile", "peuplier", "hanche", "chronomètre", "isoloirs", "théorie", 
              "ethnologie", "budget", "dignité", "tendresse", "explosion", "cyclisme", "ablation"]

print("Start to load the Data ")
e = Embeddings.load(generalDataPath + "Embeddings/vecs.npy")
X_test = np.load(fileToTestPath + "_X.npy")
y_test = np.load(fileToTestPath + "_y.npy")
print("End to load the Data")

print("The size of Test Set is : " + str(len(X_test)) + "\n \n")

###############################################################################

def getLvlOfCompositionalityByMean(word,index): # Moyenne de nos deux embeddings
    n1n2Vect = X_test[index]
    n1Vect = n1n2Vect[:300]
    n2Vect = n1n2Vect[300:]
    n1Vect = np.multiply(n1Vect,1/2)
    n2Vect = np.multiply(n2Vect,1/2)
    n1N2Vect = n1Vect + n2Vect # 1/2 * N1 + 1/2 * N2
    N1deN2Test = y_test[index] # The array to compare
    
    compositionality = tf.keras.losses.cosine_similarity(n1N2Vect,N1deN2Test)
    compositionality = compositionality.numpy()

    return (word,compositionality)

###############################################################################

def getLvlOfCompositionalityN1(word,index):
    n1n2Vect = X_test[index]
    n1Vect = n1n2Vect[:300]

    N1deN2Test = y_test[index] # The array to compare
    
    compositionality = tf.keras.losses.cosine_similarity(n1Vect,N1deN2Test)
    compositionality = compositionality.numpy()

    return (word,compositionality)

###############################################################################

def getLvlOfCompositionalityN2(word,index):
    n1n2Vect = X_test[index]
    n2Vect = n1n2Vect[300:]

    N1deN2Test = y_test[index] # The array to compare
    
    compositionality = tf.keras.losses.cosine_similarity(n2Vect,N1deN2Test)
    compositionality = compositionality.numpy()

    return (word,compositionality)

###############################################################################

def getWordTested():
    wordTest = []
    data = open(fileToTestPath +".txt")
    for line in data.readlines():
        wordTest.append(line)
    data.close()
    return wordTest

###############################################################################

def convertLossToPositive(loss):
    return ((-1*loss)+1)/2

###############################################################################

def getAverageOfCompositionalityOfWordTest(word,index):
    

    N1deN2Test = y_test[index] # The array to compare
    tmpAdd = 0
    for i in range(len(wordTest)):
        word = wordTest[i]
        word = word[:len(word)-1]
        vectWordTest = e._w2v[word]
        vectWord = np.zeros(shape=(1,300))
        vectWord[0] = vectWordTest
        
        compositionality = tf.keras.losses.cosine_similarity(vectWord,N1deN2Test)
        tmpAdd += convertLossToPositive(compositionality.numpy())
    
    return (word,tmpAdd/len(wordTest))

###############################################################################



def makeStatsOnTuple(listOfTuple):
    count = 0
    index = 0
    tabOfSimilarity = np.zeros(shape=len(listOfTuple))
    for elt in listOfTuple:
        tabOfSimilarity[index] = elt[1]
        index +=1
        count +=elt[1]
    print("Mean of similarity is " + str(count/len(listOfTuple)))
    print("Variance : " + str(np.var(tabOfSimilarity)))
    print(listOfTuple[0])
    print(listOfTuple[-1])

###############################################################################

def examineDataSet(dataSetVect,dataSetWord):
    listOfTuple = []
    listOfTupleN1 = []
    listOfTupleN2 = []
    for index in range(len(dataSetWord)):
        listOfTuple.append(getLvlOfCompositionalityByMean(dataSetWord[index],index))
        listOfTupleN1.append(getLvlOfCompositionalityN1(dataSetWord[index],index))
        listOfTupleN2.append(getLvlOfCompositionalityN2(dataSetWord[index],index))
    print("Mean of vect")
    makeStatsOnTuple(listOfTuple)
    print("Similarity N1")
    makeStatsOnTuple(listOfTupleN1)
    print("Similarity N2")
    makeStatsOnTuple(listOfTupleN2)
    
###############################################################################

def examineDataSetAndWrite(dataSetVect,dataSetWord):
    meanOfN1 = 0
    meanOfN2 = 0
    tabOfWordInKNN = []
    tabOfRankInKNN = []
    avgWordTest = 0
    sumRank = 0
    for index in range(len(dataSetWord)):
        meanOfN1 += convertLossToPositive(getLvlOfCompositionalityN1(dataSetWord[index],index)[1])
        meanOfN2 += convertLossToPositive(getLvlOfCompositionalityN2(dataSetWord[index],index)[1])
        avgWordTest += getAverageOfCompositionalityOfWordTest(dataSetWord[index],index)[1]
        word = dataSetWord[index]
        word = word[:len(word)-1]
        f = open(pathToExp + str(index) + "_" + word + ".txt","w")
        tabTmp = e.most_similar(word,100)
        n1Word = word.split("-")[0]
        n2Word = word.split("-")[2]
        indexRank = 1
        for elt in tabTmp:
            indexRank +=1
            if(elt[1] == n1Word or elt[1] == n2Word):
                f.writelines(elt[1] + " Rang : " + str(indexRank))
                tabOfRankInKNN.append(indexRank)
                sumRank += indexRank
                tabOfWordInKNN.append(word)
                break
            else:
                f.writelines(elt[1]+"\n")
        f.close()
    
    f = open(pathToExp + "00_SomeStats.txt","w")
    f.write("Moyenne des N1 Similarity : "  + str(meanOfN1/len(dataSetWord)) + "\n")
    f.write("Moyenne des N2 Similarity : "  + str(meanOfN2/len(dataSetWord)) + "\n")
    f.write("Moyenne des Similarité par rapport au mot fixe : " + str(avgWordTest/len(dataSetWord)) + "\n")
    f.write("Part des fois où N1 ou N2 est dans les 100 KNN : " + str(len(tabOfRankInKNN)/len(dataSetWord)) + "\n")
    f.write("Rang moyen parmis les N1 ou les N2 sont dans les 100 KNN : " + str(sumRank/100) + "\n")
    
    
    f.write(str(tabOfWordInKNN).strip('[]')+ "\n")
    f.write(str(tabOfRankInKNN).strip('[]')+ "\n")
    f.close()
        
###############################################################################

wordTest = getWordTested()
examineDataSetAndWrite(X_test,wordTest)
