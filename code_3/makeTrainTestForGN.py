#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:33:15 2021

@author: yannis.coutouly


This file is to make dataset for DET_N_ADJ we will use the representation of le la les for train
and the representation of un une des for dev and test 
"""

from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf

generalPath = "../BatchExp/ForthBatchExp/DataExperience/"

print("Load the Embeddings")
embeddingsTog = Word2Vec.load(generalPath + "Embeddings/embeddings_300_full10Fusion.model")
model = tf.keras.models.load_model(generalPath + "NeuralNetwork/Encode_N_ADJ_Train=WACFull10_Encode=300")


print("Embeddings is Loaded")


###############################################################################

def getWordInLine(line):
    word = ""
    lastWord = False
    indexLastWord = 0
    for i in range(len(line)):
        if(not(lastWord) and not(line[i] == " ")):
            lastWord = True
            indexLastWord = i
            continue
        if(lastWord and line[i] == "-"):
            word = line[indexLastWord:]
            break
        if(lastWord and line[i] == " "):
            lastWord = False
    
    word = line[indexLastWord:]
    return word

###############################################################################

def makeTabOfN_ADJ():
    f = open(generalPath + "Utils/Det_Noun_Adj_Full10Sorted5.txt")
    pseudoRandomNumber = 0
    tabTrain = []
    tabTest = []
    tabDev = []
    for line in f.readlines():
        lineWord = getWordInLine(line)

        if(pseudoRandomNumber % 20 < 14):
            tabTrain.append(lineWord)
        elif(pseudoRandomNumber % 20 < 17):
            tabDev.append(lineWord)
        else:
            tabTest.append(lineWord)
        if(len(tabTest) > 1000):
            break
        pseudoRandomNumber +=1
    f.close()
    return tabTrain,tabDev,tabTest


###############################################################################

def writeDataInFile(tabData,path):
    writeFile = open(path+".txt","w")
    npTab = np.zeros(shape=(len(tabData),600))
    nbrProblem = 0
    for index in range(len(tabData)):
        if(index % 1000 == 0):
            print(str(index) + " sur " + str(len(tabData)))
        data = tabData[index]
        data = data[:len(data)-1]
        word1 = data.split("-")[0]
        word2 = data.split("-")[1]
        word3 = data.split("-")[2]
        if(not(word1 in embeddingsTog.wv)):
            print(word1)
            nbrProblem += 1
            continue
        if(not(word2 in embeddingsTog.wv)):
            print(word2)
            nbrProblem += 1
            continue
        if(not(word3 in embeddingsTog.wv)):
            print(word3)
            nbrProblem += 1
            continue
        vectWord1 = embeddingsTog.wv[word1]
        vectWord2 = embeddingsTog.wv[word2]
        vectWord3 = embeddingsTog.wv[word3]
        concatenate = np.zeros(shape=(1,600))
        concatenate[0] = np.concatenate((vectWord2, vectWord3))
        predicted = model.predict(concatenate) # The N_ADJ Space
        concatenate[0] = np.concatenate((vectWord1,predicted[0])) # Det and N_ADJ
        npTab[index] = concatenate
        writeFile.write(data + "\n")
    writeFile.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(npTab)))
    for i in range(nbrProblem):
        if(i % 100 == 0):
            print(i)
        npTab = np.delete(npTab,len(tabData)-i-1,0)
    
    np.save(path + ".npy",npTab)

###############################################################################

tabTrain, tabDev, tabTest = makeTabOfN_ADJ()    

print("Tab are made")

#writeDataInFile(tabTrain,generalPath  + "Train/train_DET_N_ADJ_Full")
#writeDataInFile(tabDev,generalPath  + "Dev/dev_DET_N_ADJ_Full")
writeDataInFile(tabTest,generalPath  + "Test/test_DET_N_ADJ_1000")