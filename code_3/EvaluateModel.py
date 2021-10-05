#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:09:24 2021

@author: yannis.coutouly

In this file we will evaluate our model with the use of a TOP10
1/rank then the mean 

"""
from gensim.models import Word2Vec
import numpy as np
import math
import tensorflow as tf

#Constant Variable ------------------------------------------------------------------------------------

generalDataPath = "../BatchExp/ForthBatchExp/DataExperience/"
testDataPath = generalDataPath+"Test"
neuralNetworkPath = generalDataPath + "NeuralNetwork"

resultExperimentPath = "../BatchExp/ForthBatchExp/Experience/Exp_N_ADJ_EPS2_Train=little_Encode300_Test=1000/"
modelPath = neuralNetworkPath + "/NN_N_ADJ_V2_lossV2Adjusted"
typeOfTest = "/test_N_ADJ_1000"

#Load Data ---------------------------------------------------------------------------------------
print("Start to load the Data ")
X_test = np.load(testDataPath + typeOfTest + ".npy")
arrDev = np.split(X_test,[300],axis=1)
word1_Test = arrDev[0]
word2_Test = arrDev[1]
model = tf.keras.models.load_model(modelPath)
embeddings = Word2Vec.load(generalDataPath+"Embeddings/embeddings_300_full10Fusion.model")
print("End to load the Data")

print("The size of Test Set is : " + str(len(X_test)) + "\n \n")
#Eval Model --------------------------------------------------------------------------------------

def computeTopK():
    f = open(testDataPath + typeOfTest + ".txt")
    index = 0
    sumRankWord1 = 0 
    sumRankWord2 = 0 
    for data in f.readlines():
        if(index % 100 == 0):
            print(index)
        line = data[:len(data)-1]
        word1 = line.split("-")[0]
        word2 = line.split("-")[1]
        vectData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectData[0] = X_test[index]
        predict = model.predict(vectData)
        neighborsWord1 = embeddings.wv.most_similar(predict[0], topn=10)
        neighborsWord2 = embeddings.wv.most_similar(predict[1], topn=10)
        for i in range(11):
            if(i == 10):
                sumRankWord1 += 0
                continue
            if(neighborsWord1[i][0] == word1):
                sumRankWord1 += 1/(i+1)
                break
        for i in range(11):
            if(i == 10):
                sumRankWord2 += 0
                continue
            if(neighborsWord2[i][0] == word2):
                sumRankWord2 += 1/(i+1)
                break
        index+=1
    print("Moyenne des ranks des N1 : " + str(sumRankWord1/index))
    print("Moyenne des ranks des N2 : " + str(sumRankWord2/index))
    
def computeTopKForGN():
    scoreDet = 0
    f = open(testDataPath + typeOfTest + ".txt")
    for index in range(len(X_test)):
        if(index % 100 == 0):
            print(index)
        vectData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectData[0] = X_test[index]
        predict = model.predict(vectData)
        neighborsDet = embeddings.wv.most_similar(predict[1], topn=10) #Get the Det prediction
        line = f.readline()
        det = line.split("-")[0]
        for i in range(len(neighborsDet) + 1):
            if(i == len(neighborsDet)):
                scoreDet += 0
                break
            if(neighborsDet[i][0] == det):
                scoreDet += 1/(i+1)
                break
    print("Moyenne des ranks des Det : " + str(scoreDet/len(X_test)))
                
    
def computeTopKForV2():
    f = open(testDataPath + typeOfTest + ".txt")
    index = 0
    sumRankWord1 = 0 
    sumRankWord2 = 0 
    for data in f.readlines():
        if(index % 100 == 0):
            print(index)
        line = data[:len(data)-1]
        word1 = line.split("-")[0]
        word2 = line.split("-")[1]
        vectData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectData[0] = X_test[index]
        tupleSplit = np.split(vectData,[300],axis=1)
        word1_Data = tupleSplit[0]
        word2_Data = tupleSplit[1]
        predict = model.predict((vectData,word2_Data,word1_Data,vectData,word2_Data,word1_Data))
        neighborsWord1 = embeddings.wv.most_similar(predict[0], topn=10)
        neighborsWord2 = embeddings.wv.most_similar(predict[1], topn=10)
        for i in range(11):
            if(i == 10):
                sumRankWord1 += 0
                continue
            if(neighborsWord1[i][0] == word1):
                sumRankWord1 += 1/(i+1)
                break
        for i in range(11):
            if(i == 10):
                sumRankWord2 += 0
                continue
            if(neighborsWord2[i][0] == word2):
                sumRankWord2 += 1/(i+1)
                break
        index+=1
    print("Moyenne des ranks des N1 : " + str(sumRankWord1/index))
    print("Moyenne des ranks des N2 : " + str(sumRankWord2/index))
computeTopKForV2()