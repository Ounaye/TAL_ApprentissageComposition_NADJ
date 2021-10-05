#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:21:31 2021

@author: yannis.coutouly

This file is for extract the embeddings of the N_ADJ form
This will speed Up the use of our NN

We want to produce an 600 dim tab ( concatenate our 2 )
The input and the output are the same so only X_[test,train,dev] file are required
We want a 70/15/15 repartition

"""
from gensim.models import Word2Vec
import numpy as np

generalPath = "../BatchExp/ThirdBatchExp/DataExperience/"

print("Load the Embeddings")
embeddingsTog = Word2Vec.load(generalPath+"Embeddings/embeddings_300_full10Fusion.model")
print("Embeddings is Loaded")


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

def makeTabOfN_ADJ():
    f = open(generalPath + "Utils/N_ADJ_Full10ReduceSimple.txt")
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
        pseudoRandomNumber +=1
    f.close()
    return tabTrain,tabDev,tabTest
        
    
def writeDataInFile(tabData,path):
    writeFile = open(path+".txt","w")
    npTab = np.zeros(shape=(len(tabData),600))
    nbrProblem = 0
    for index in range(len(tabData)):
        data = tabData[index]
        data = data[:len(data)-1]
        word1 = data.split("-")[0]
        word2 = data.split("-")[1]
        if(not(word1 in embeddingsTog.wv)):
            print(word1)
            nbrProblem += 1
            continue
        if(not(word2 in embeddingsTog.wv)):
            print(word2)
            nbrProblem += 1
            continue
        vectWord1 = embeddingsTog.wv[word1]
        vectWord2 = embeddingsTog.wv[word2]
        concatenate = np.concatenate((vectWord1, vectWord2))
        npTab[index] = concatenate
        writeFile.write(data + "\n")
    writeFile.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(npTab)))
    for i in range(nbrProblem):
        if(i % 100 == 0):
            print(i)
        npTab = np.delete(npTab,len(tabData)-i-1,0)
    
    np.save(path + ".npy",npTab)

def writeDataInFileAndAddEpsilon(tabData,path,N):
    writeFile = open(path+".txt","w")
    npTab = np.zeros(shape=(len(tabData),600))
    nbrProblem = 0
    for index in range(len(tabData)):
        data = tabData[index]
        data = data[:len(data)-1]
        word1 = data.split("-")[0]
        word2 = data.split("-")[1]
        if(not(word1 in embeddingsTog.wv)):
            print(word1)
            nbrProblem += 1
            continue
        if(not(word2 in embeddingsTog.wv)):
            print(word2)
            nbrProblem += 1
            continue
        if(index % N == 0): # We erase the ADJ part 
            vectWord1 = embeddingsTog.wv[word1]
            concatenate = np.concatenate((vectWord1, vectWord1))
            npTab[index] = concatenate
            writeFile.write(data + "-EPS\n")
            continue 
        vectWord1 = embeddingsTog.wv[word1]
        vectWord2 = embeddingsTog.wv[word2]
        concatenate = np.concatenate((vectWord1, vectWord2))
        npTab[index] = concatenate
        writeFile.write(data + "\n")
    writeFile.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(npTab)))
    for i in range(nbrProblem):
        if(i % 100 == 0):
            print(i)
        npTab = np.delete(npTab,len(tabData)-i-1,0)
    
    np.save(path + ".npy",npTab)

tabTrain, tabDev, tabTest = makeTabOfN_ADJ()    

writeDataInFile(tabTrain,generalPath  + "Train/train_DET_N_ADJ_Full")
writeDataInFile(tabDev,generalPath  + "Dev/dev_DET_N_ADJ_Full")
writeDataInFile(tabTest,generalPath  + "Test/test_DET_N_ADJ_Full")
