#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:16:56 2021

@author: yannis.coutouly
"""

"""
This file have for goal to eval the level of compositionality of a dataset

On prend l'embeddings N1 de N2 et on fait la somme de notre cosine similarity
L'idée est de voir si il y en a qui ont une valeur vraiment désastreuse pour les enlever du training set 
Dans le futur ce programme pourras être utile en tant que gatekeeper pour le full random test

"""


import random as rd
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

#Constant Variable ------------------------------------------------------------------------------------

generalDataPath = "../BatchExp/FirstBatchExp_VPropre/DataExperience/"
fileToTestPath = generalDataPath + "Test/test_N1=8but10_N1=5"
pathToExp = "../BatchExp/SecondBatchExp/Experience/Compos_test_N1=8but10_N1=5/"

fixedNouns = ["chiot", "agriculteur", "étoile", "peuplier", "hanche", "chronomètre", "isoloirs", "théorie", 
              "ethnologie", "budget", "dignité", "tendresse", "explosion", "cyclisme", "ablation"]


#Load Data ---------------------------------------------------------------------------------------
print("Start to load the Data ")
embeddings = Word2Vec.load(generalDataPath+"Embeddings/embeddings_300_full10Fusion.model")
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

def getAverageOfCompositionalityOfWordTest(word,index):
    

    N1deN2Test = y_test[index] # The array to compare
    tmpAdd = 0
    for i in range(len(wordTest)):
        word = wordTest[i]
        word = word[:len(word)-1]
        vectWordTest = embeddings.wv[word]
        vectWord = np.zeros(shape=(1,300))
        vectWord[0] = vectWordTest
        
        compositionality = tf.keras.losses.cosine_similarity(vectWord,N1deN2Test)
        tmpAdd += convertLossToPositive(compositionality.numpy())
    
    return (word,tmpAdd/len(wordTest))

###############################################################################

def convertLossToPositive(loss):
    return ((-1*loss)+1)/2

###############################################################################
def getWordTested():
    wordTest = []
    data = open(fileToTestPath +".txt")
    for line in data.readlines():
        wordTest.append(line)
    data.close()
    return wordTest

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
    avgWordTest = 0
    tabOfWordInKNN = []
    tabOfRankInKNN = []
    sumRank = 0
    for index in range(len(dataSetWord)):
        meanOfN1 += convertLossToPositive(getLvlOfCompositionalityN1(dataSetWord[index],index)[1])
        meanOfN2 += convertLossToPositive(getLvlOfCompositionalityN2(dataSetWord[index],index)[1])
        avgWordTest += getAverageOfCompositionalityOfWordTest(dataSetWord[index],index)[1]
        word = dataSetWord[index]
        word = word[:len(word)-1]
        f = open(pathToExp + str(index) + "_" + word + ".txt","w")
        tabTmp = embeddings.wv.most_similar([word], topn=100)
        n1Word = word.split("-")[0]
        n2Word = word.split("-")[2]
        indexRank = 1
        for elt in tabTmp:
            indexRank +=1
            if(elt[0] == n1Word or elt[0] == n2Word):
                f.writelines(elt[0] + " Rang : " + str(indexRank))
                tabOfRankInKNN.append(indexRank)
                sumRank += indexRank
                tabOfWordInKNN.append(word)
                break
            else:
                f.writelines(elt[0]+"\n")
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

#wordTest = getWordTested()
#examineDataSetAndWrite(X_test,wordTest)
print("salle \n")
print(embeddings.wv.most_similar(["salle"], topn=100))

print("\n \n bain \n")
print(embeddings.wv.most_similar(["bain"], topn=100))