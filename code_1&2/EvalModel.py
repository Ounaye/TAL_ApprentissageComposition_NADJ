#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:00:49 2021

@author: yannis.coutouly
"""

import random as rd
import numpy as np
import math
import tensorflow as tf

#Constant Variable ------------------------------------------------------------------------------------

generalDataPath = "../BatchExp/SecondBatchExp/DataExperience/"
testDataPath = generalDataPath+"Test"
neuralNetworkPath = generalDataPath + "NeuralNetwork"

resultExperimentPath = "../BatchExp/SecondBatchExp/Experience/Exp_300N1_300N2_W2VF_Test=N1300_N2300/"
modelPath = neuralNetworkPath + "/NN_Hide=480_Train=_300N1_300N2_W2VF"
typeOfTest = "/test_300N1_300N2"

allN1 =  ["salle", "outil","jeu","règle","équipe","chef","liste","acte","zone","technique","question",
                "méthode", "droit"]
allN1Except2 = ["salle", "outil","jeu","règle","équipe","chef","liste","zone","question",
                "méthode", "droit"]

allN1Except5 = ["salle", "outil","jeu","règle","équipe","chef","liste","zone"]

trainSet = allN1Except2

print("On va tester " + typeOfTest[1:] + "sur le jeu de donnée : " + "allN1Except2")

#Load Data ---------------------------------------------------------------------------------------
print("Start to load the Data ")
X_test = np.load(testDataPath + typeOfTest + "_X.npy")
y_test = np.load(testDataPath + typeOfTest + "_y.npy")
model = tf.keras.models.load_model(modelPath)
print("End to load the Data")

print("The size of Test Set is : " + str(len(X_test)) + "\n \n")
#Eval Model --------------------------------------------------------------------------------------

def evalModel_2(X_test,y_test,wordTest,writeResult = True): # 1 minutes de calcul
    listOfRank = []
    index = 0
    for i in range(len(X_test)): #For all N1deN2
        vectXData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectXData[0] = X_test[i]
        vectPredict = model.predict(vectXData)
        vectPredict = np.float64(vectPredict) #For CosineSimilarity
        
        #The level of Compositionality :
        n1Similarity, n2Similarity = getLvlOfCompositionality(wordTest[i],i)
        
        listOfTuple = makeListOfTuple(vectPredict,wordTest)
        listOfRank.append((getRank(wordTest[i], listOfTuple),wordTest[i],n1Similarity,n2Similarity))
        #Write in the file the result ---------------------------------------
        if(index % 25 == 0):
            print(index)
        index += 1
        if(writeResult):       
            f = open(resultExperimentPath+str(listOfRank[-1][0])+"_"+wordTest[i], "w")
            f.write("Niveau de compositionalité N1 :\t" + str(n1Similarity)+ "\tN2 :\t" + str(n2Similarity) + "\n")
            for line in listOfTuple:
                f.write(str(line[0])+"\t"+line[1])
                if(line[1]==wordTest[i]):
                    f.write("\tRank : " + str(listOfRank[-1][0]))
                f.write("\n")
            f.close()
        
    return listOfRank
            

def makeListOfTuple(n1Vect,wordTest):
    listOfTuple = [] 
    for j in range(len(y_test)): #Look at all N1-de-N2
            vectToTest = np.zeros(shape=(1,300))
            vectToTest[0] = y_test[j]
            accuracy = tf.keras.losses.cosine_similarity(vectToTest,n1Vect)
            accuracy = accuracy.numpy()[0]
            wordTested = wordTest[j]
            tupleToAdd = (accuracy,wordTested)
            tupleGotInsered = False
            #On peut le faire en O(log(n))
            if(len(listOfTuple) == 0):
                listOfTuple.insert(0,tupleToAdd)
            else:
                indexToInsert = findByDichotomie(0,len(listOfTuple),accuracy,listOfTuple)
                listOfTuple.insert(indexToInsert,tupleToAdd)
            """
            for k  in range(len(listOfTuple)): # Insert them in the right place in the list
                if(accuracy < listOfTuple[k][0]): #Find somewhere to insert in the right order 
                    listOfTuple.insert(k, tupleToAdd)
                    tupleGotInsered = True
                    break
            if(not tupleGotInsered):
                listOfTuple.append(tupleToAdd)
            """
    return listOfTuple

def getRank(word,listOfTuple):
    for i in range(len(listOfTuple)):
        if(listOfTuple[i][1] == word):
            return i
    return -1

def findByDichotomie(indexStart,indexEnd,elt,listOfObject):
    if(indexEnd - indexStart < 2):
        if(listOfObject[indexStart][0] == elt):
            return indexStart
        else: 
            return indexStart
    diff = indexEnd - indexStart
    newIndex = indexStart +  diff/2
    newIndex = math.floor(newIndex)
    if(elt < listOfObject[newIndex][0]):
        return findByDichotomie(indexStart, newIndex, elt, listOfObject)
    else:
        return findByDichotomie(newIndex, indexEnd, elt, listOfObject)
        

def getLvlOfCompositionality(word,index):
    n1n2Vect = X_test[index]
    n1Vect = n1n2Vect[:300]
    n2Vect = n1n2Vect[300:]
    N1deN2Test = y_test[index]
    
    n1Similarity = tf.keras.losses.cosine_similarity(n1Vect,N1deN2Test)
    n2Similarity = tf.keras.losses.cosine_similarity(n2Vect,N1deN2Test)
    n1Similarity = n1Similarity.numpy()
    n2Similarity = n2Similarity.numpy()
    return n1Similarity,n2Similarity

def meanOfN1N2Similarity(X_test,y_test,wordTest):
    meanOfN1 = 0
    meanOfN2 = 0
    for i in range(len(X_test)): #For all N1deN2
        vectXData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectXData[0] = X_test[i]
        vectPredict = model.predict(vectXData)
        vectPredict = np.float64(vectPredict) #For CosineSimilarity
        
        #The level of Compositionality :
        n1Similarity, n2Similarity = getLvlOfCompositionality(wordTest[i],i)
        meanOfN1 += (n1Similarity+1)
        meanOfN2 += (n2Similarity+1)
    meanOfN1 = meanOfN1/(2*len(X_test))
    meanOfN2 = meanOfN2/(2*len(X_test))
    print("Dumb Mean N1 : " + str(meanOfN1))
    print("Dumb Mean N2 : " + str(meanOfN2))



def getWordTested():
    wordTest = []
    data = open(testDataPath + typeOfTest +".txt")
    for line in data.readlines():
        wordTest.append(line)
    data.close()
    return wordTest
        
    

def meanOfRank(rankList):
    sumRank = 0
    for rank in rankList:
        sumRank += rank[0]
    sumRank = sumRank/len(rankList)
    return sumRank

def simpleMeanReciprocalRank(rankList):
    add = 0
    for nbr in rankList:
        add += 1/(nbr[0]+1)
    add = add*(1/len(rankList))
    return add
    
def N1ClassMRR(rankList): #Class between N1 seen and N1 not seen in the train
    clN1 = 0
    nbrN1 = 0
    clOther = 0
    nbrOther = 0
    for tpl in rankList:
        wordN1 = tpl[1].split("-")[0]
        if(not(wordN1 in trainSet)): # Find the N1 who arent in the train data
            clN1 += 1/(tpl[0]+1)
            nbrN1+=1
        else:
            clOther += 1/(tpl[0]+1)
            nbrOther+=1
    if(nbrN1 == 0 or nbrOther == 0):
        return (-1,-1)
    mrrN1 = clN1*(1/nbrN1)
    mrrOther = clOther*(1/nbrOther)
    return (mrrN1,mrrOther)
        
def N1ToN2ClassMRR(rankList):
    clN1 = 0
    nbrN1 = 0
    clN2 = 0
    nbrN2 = 0
    for tpl in rankList:
        if(tpl[2] < tpl[3]):#N1 < N2
            clN1 += 1/(tpl[0]+1)
            nbrN1+=1
        else:
            clN2 += 1/(tpl[0]+1)
            nbrN2+=1
    mrrN1 = clN1*(1/nbrN1)
    mrrN2 = clN2*(1/nbrN2)
    return (mrrN1,mrrN2)



def sortByN1AndN2(rankList):
    rankList.sort(key = lambda x : 1/2*x[2]+1/2*x[3])
    listRank = []
    for val in rankList:
        listRank.append(val[0])
    return listRank

def writeTheResult(rankList):
     f = open(resultExperimentPath+"_00SomeStats", "w")
     
     f.write("The Mean of all the rank is : " + str(meanOfRank(rankList)) + "\n")
     f.write("The Mean reciprocal rank : " + str(simpleMeanReciprocalRank(rankList)) + "\n")
     tupleMrrN1 = N1ClassMRR(rankList)
     if(tupleMrrN1 == (-1,-1)):
         f.write("No new N1 in the Test Set \n")
     else:
        f.write("The Mean reciprocal rank with N1NotSeen: " + str(tupleMrrN1[0]) + "\n")
        f.write("The Mean reciprocal rank with N1Seen: " + str(tupleMrrN1[1]) + "\n")
     tupleN1N2MRR = N1ToN2ClassMRR(rankList)
     f.write("MRR of N1<N2 " + str(tupleN1N2MRR[0]) + "\n")
     f.write("MRR of N2<N1 " + str(tupleN1N2MRR[1]) + "\n")
     rankListSorted = sortByN1AndN2(rankList)
     f.write(str(rankListSorted).strip('[]'))
     

wordTest = getWordTested()
#meanOfN1N2Similarity(X_test,y_test,wordTest)
rankList = evalModel_2(X_test,y_test,wordTest,True)

writeTheResult(rankList)

