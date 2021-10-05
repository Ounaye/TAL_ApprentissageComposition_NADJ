#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:56:43 2021

@author: yannis.coutouly


Look what a new space look like 
We will use the Norme_1 and ask whe N closest point of on point

One function of distance
One function of who take a pts, a space ( an array), and N
    Ang find the N closest pts in the space

"""
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from scipy import linalg, mat, dot
import math


generalDataPath = "../BatchExp/ForthBatchExp/DataExperience/"
dataToTest = generalDataPath + "Test/test_N_ADJ_Full"




#Load Data --------------------------------------------------------------------------------------

print("Load the data")
"""
testData = np.load(dataToTest + ".npy")
arrDev = np.split(testData,[300],axis=1)
word1_vect = arrDev[0]
word2_vect = arrDev[1]
"""
testData = np.load(dataToTest + ".npy")






model = tf.keras.models.load_model(generalDataPath + "NeuralNetwork/Encode_N_ADJ_V2_lossV2Adjusted")
tabSpace = model.predict(testData)



print("Data are loaded")

###############################################################################

def makeTabString():
    tab = []
    f = open(dataToTest + ".txt")
    for line in f.readlines():
        tab.append(line[:len(line)-1])
    return tab

###############################################################################

def cosSimilarity(x,y):
    return  np.dot(x,y.T)/linalg.norm(x)/linalg.norm(y)


def Norme_1(x,y):
    tmp = 0
    for i in range(len(x)):
        tmp += abs(x[i]-y[i])
    return tmp

def getNClosestPts(pts,space,N):
    f = open(dataToTest + ".txt")
    tabOfClosest = []
    for i in range(len(space)):
        tmpTxt = f.readline()
        tmpNorm = cosSimilarity(pts,space[i])
        tabOfClosest.append((tmpNorm,tmpTxt))
        
    tabOfClosest.sort(key=lambda x : x[0])
    tabOfClosest.reverse()
    return tabOfClosest[:N-1]

###############################################################################

"""
Complexity : MAX[O(N),O(m*log(m))]

N : Data in the dataTest
m : Data with the word 

Inputs :

elt : a string of NOUN-ADJ

This fonction is for detecting a cluster of a word
It will return a tuple of the NOUN cluster and the ADJ cluster [O(N)]
The algo have to find every occurence of the NOUN and the ADJ 
Make a copy of all the vector  (getAlloccurence)
Then find the center of all the vect by using the mean.  [O(m)] (findCenterOfMass)
Then get the Norm1, sort the tab and get the 90 % better [O(m*log(m))]
We can now get the value of the distance

"""

def getAllOccurence(word1,word2):
    tabWord1 = []
    tabWord2 = []
    for i in range(len(tabDataText)):
        line = tabDataText[i]
        word_1 = line.split("-")[0]
        word_2 = line.split("-")[1]
        if(word1 == word_1):
            tabWord1.append(tabSpace[i])
        if(word2 == word_2):
            tabWord2.append(tabSpace[i])
    return (tabWord1,tabWord2)
        
    return []

def findCenterOfMass(tabData):
    centerVect = np.zeros(shape=(1,len(tabData[0])))
    for elt in tabData:
        centerVect[0] += elt
    return centerVect[0]/len(centerVect[0])

def getCluster(elt,index):
    word1 = elt.split("-")[0]
    word2 = elt.split("-")[1]
    tabWord1, tabWord2 = getAllOccurence(word1,word2)
    centerVector_1 = findCenterOfMass(tabWord1)
    centerVector_2 = findCenterOfMass(tabWord2)
    
    tabDistWordCenter_1 = []
    tabDistWordCenter_2 = []
    
    for i in range(len(tabWord1)):
        tabDistWordCenter_1.append(cosSimilarity(centerVector_1, tabWord1[i]))
    for i in range(len(tabWord2)):
        tabDistWordCenter_2.append(cosSimilarity(centerVector_2, tabWord2[i]))
    
    tabDistWordCenter_1.sort(key=lambda x : x)
    tabDistWordCenter_2.sort(key=lambda x : x)
    
    return (cosSimilarity(centerVector_1,tabSpace[index]),cosSimilarity(centerVector_2,tabSpace[index]),
            tabDistWordCenter_1[:math.floor(len(tabDistWordCenter_1)*0.9)],tabDistWordCenter_2[:math.floor(len(tabDistWordCenter_2)*0.9)])

###############################################################################

def printInfoClusterElt(tabDataText,index):
    print(tabDataText[index])
    word1Center, word2Center, clusterWord1,clusterWord2 = getCluster(tabDataText[index],index)
    if(clusterWord1 == [] or clusterWord2 == []):
            return
    print("Distance to Center NOUN: " + str(word1Center) + " ADJ: " + str(word2Center))
    print("Distance max NOUN: " + str(clusterWord1[len(clusterWord1)-1]) + " ADJ: " + str(clusterWord2[len(clusterWord2)-1]))


def makeStatsOnCluster(tabData): # O[N*N]
    clusterWord1Tmp = 0
    clusterWord2Tmp = 0
    for i in range(len(tabData)):
        if( i % 1000 == 0):
            print(i)
        word1Center, word2Center, clusterWord1,clusterWord2 = getCluster(tabDataText[i],i)
        if(word1Center < 0.98):
            print("Error : " + str(word1Center) + tabDataText[i] )
        if(clusterWord1 == [] or clusterWord2 == []):
            continue
        clusterWord1Tmp += clusterWord1[len(clusterWord1)-1]
        clusterWord2Tmp += clusterWord2[len(clusterWord2)-1]
    print("La taille moyenne des clusters pour les NOUNS est " + str(clusterWord1Tmp/len(tabData)))
    print("La taille moyenne des clusters pour les ADJ est " + str(clusterWord2Tmp/len(tabData)))
###############################################################################
azd = 22222
tabDataText = makeTabString()
"""

print("For the new space :\n \n")
print(getNClosestPts(tabSpace[0][azd],tabSpace[0],10))

"""
#print("\n \nIn the old space : \n \n")
#print(getNClosestPts(tabSpace[azd],tabSpace,25))



makeStatsOnCluster(tabSpace)

        
        
        
        
        
        
        
        
        
        