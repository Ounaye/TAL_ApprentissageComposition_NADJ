#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:00:29 2021

@author: yannis.coutouly
"""

from gensim.models import Word2Vec


import tensorflow as tf
import numpy as np
import math
from keras.models import Sequential
from keras.layers import InputLayer, Dense

#Load data and construct some representation-------------------------------------------------------

#Select the word we want to work with
tabOfN1Taken = ["salle", "outil","jeu","règle","équipe","chef","liste","acte","zone","technique","question",
                "méthode", "droit"]
mapOfN1andN2 = dict([("salle", []),
                     ("outil", []),
                     ("jeu", []),
                     ("règle", []),
                     ("équipe", []),
                     ("chef", []),
                     ("liste", []),
                     ("acte", []),
                     ("zone", []),
                     ("technique", []),
                     ("question", []),
                     ("méthode", []),
                     ("droit", [])])

tabOfN1ForTest = ["acte","technique"]

def isInTheN1Taken(word):
    for wordN1 in tabOfN1Taken:
        if(wordN1 == word):
            return True
    return False

def makedictOfN1andN2():
    data = open("./fileProcess/MakingEmbeddings/N1deN2SortedLemmeTabCut.txt")
    for line in data.readlines():
        #line = line.decode("utf-8")
        n1 = line.split("\t")[0]
        for index in range(len(n1)):
            if(n1[index] == " " or n1[index].isdigit()):
                continue
            else:
                n1 = n1[index:]#Get only the word not the number of occurence
                break
        
        n2 = line.split("\t")[2]
        n2 = n2[:len(n2)-1] #Remove the space at the end
        if(isInTheN1Taken(n1)):    
            mapOfN1andN2[n1].append(n2)


makedictOfN1andN2()

# Construct the Input and Output vector------------------------------------------------------


print("make the input and the output")


# On évite de charger les embeddings à chaque fois

def findSizeTabInputOutput():
    sumN1Test = 0
    sumOtherN1 = 0
    tmpOfN1 = 0
    for n1Word in tabOfN1Taken:
        if(n1Word in tabOfN1ForTest):#N1 for Test
            tabOfN2 = mapOfN1andN2.get(n1Word)
            sumN1Test += len(tabOfN2)
        else:
            tabOfN2 = mapOfN1andN2.get(n1Word)
            tmpOfN1 += len(tabOfN2)
    sumN1Test += 0.1 * tmpOfN1 + len(tabOfN1Taken) - len(tabOfN1ForTest)
    sumN1Test = math.floor(sumN1Test) +1
    sumOtherN1 += 0.9 * tmpOfN1 - len(tabOfN1Taken) + len(tabOfN1ForTest)
    return sumN1Test,sumOtherN1
        

def makeInputOutputTrainTestVector():
    
    print("Load the Embeddings")
    modelW2C = Word2Vec.load("./model/Embeddings_V2/BigWor2VecN1DeN2_300_v2.model")
    modelW2CResult =  Word2Vec.load("./model/Embeddings_V2/BigWor2VecN1DeN2Together_300_v2.model")
    #This is a determinist method to make the tab, the data are still saved somewhere in case of problem
    sizeTrain, sizeTest = findSizeTabInputOutput()
    #Automatiser le fait de trouver la taille 
    trainInput = np.zeros(shape=(sizeTrain,600))
    trainOutput = np.zeros(shape=(sizeTrain,300))
    testInput = np.zeros(shape=(sizeTest,600))
    testOutput= np.zeros(shape=(sizeTest,300))
    
    indexTrain = 0
    indexTest = 0
    pseudoRandomDistr = 0
    for n1Word in tabOfN1Taken:
        tabOfN2 = mapOfN1andN2.get(n1Word)
        vectOfN1 = modelW2C.wv[n1Word]
        
        for n2Word in tabOfN2:
            vectOfN2 = modelW2C.wv[n2Word]
            n1n2Word = n1Word + "-de-" + n2Word #Get the N1-de-N2 form
            vectN1N2 = np.hstack([vectOfN1,vectOfN2]) # Make the input vector
            vectN1deN2 = modelW2CResult.wv[n1n2Word]  # Make the output vector
            if(n1Word in tabOfN1ForTest):
                testInput[indexTest] = vectN1N2
                testOutput[indexTest] = vectN1deN2
                indexTest+=1
                continue
            if(pseudoRandomDistr == 9):
                pseudoRandomDistr = 0
                testInput[indexTest] = vectN1N2
                testOutput[indexTest] = vectN1deN2
                indexTest+=1
            else:
                trainInput[indexTrain] = vectN1N2
                trainOutput[indexTrain] = vectN1deN2
                indexTrain+=1
            pseudoRandomDistr+=1
    return trainInput,trainOutput,testInput,testOutput

#Load the array, if you change the file change the makeInputOutputTrainTestWord function
# This work because all of this work in a deterministic way
"""
a,b,c,d  = makeInputOutputTrainTestVector()

np.save("./fileProcess/DataForLearning/v2/trainInput.npy",a)
np.save("./fileProcess/DataForLearning/v2/trainOutput.npy",b)
np.save("./fileProcess/DataForLearning/v2/testInput.npy",c)
np.save("./fileProcess/DataForLearning/v2/testOutput.npy",d)

"""

print(findSizeTabInputOutput())

X_train = np.load("./fileProcess/DataForLearning/v2/trainInput.npy")
y_train = np.load("./fileProcess/DataForLearning/v2/trainOutput.npy")
X_test = np.load("./fileProcess/DataForLearning/v2/testInput.npy")
y_test = np.load("./fileProcess/DataForLearning/v2/testOutput.npy")



def makeInputOutputTrainTestWord(): #This function is build to match the file load upside
    wordTrain = []
    wordTest = []
    pseudoRandomDistr = 0
    for n1Word in tabOfN1Taken:
        tabOfN2 = mapOfN1andN2.get(n1Word)
        for n2Word in tabOfN2:
            n1n2Word = n1Word + "-de-" + n2Word #Get the N1-de-N2 form
            if(n1Word in tabOfN1ForTest):
                wordTest.append(n1n2Word)
                continue
            if(pseudoRandomDistr == 9):
                pseudoRandomDistr = 0
                wordTest.append(n1n2Word)
            else:
                wordTrain.append(n1n2Word)
            pseudoRandomDistr+=1
    return wordTrain,wordTest

#Convert To tensor

X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

"""

Pour chaque "N1 de N2" ( donc chaque elt de X_test)
    On calcul l'embeddings de "N1 de N2"
    On regarde de quel embeddings il est le plus proche:
            On fait une liste de tuple
            On parcours la liste c'est c'est plus petit on avance sinon on l'insère avant'
    On conserve le rang de "N1-de-N2" pour faire l'évalution MRR
    On stock dans un fichier texte les informations

On peut pas le faire pour l'instant car on a pas la correspondance embeddings-texte'
On aimerai afficher la similarité entre N1 et N2 sur la première ligne


"""

wordTrain, wordTest = makeInputOutputTrainTestWord()

print("The input and Output Vector were made")



#Construct the model------------------------------------------------------------------------------

#Construct Neural Network
"""
model = Sequential()
model.add(InputLayer(input_shape=(600,)))
model.add(Dense(450, activation='relu'))
model.add(Dense(300, activation='tanh'))

model.build()

model.summary()

model.compile(optimizer='adam', 
    loss="cosine_similarity")



model.fit([[X_train]],[[y_train]],epochs=30)
model.save("./model/model_V2")
"""

print("Start to load the  Model ")
model = tf.keras.models.load_model("./model/model_V2")
print("End to load the Model")
# The function for evaluating the model ----------------------------------------------------------------


import random as rd




def evalModel_2(X_test,y_test,wordTest): # 1 minutes de calcul
    listOfRank = []
    for i in range(len(X_test)): #For all N1deN2
        vectXData = np.zeros(shape=(1,600)) #To force the array to be in the good shape
        vectXData[0] = X_test[i]
        vectPredict = model.predict(vectXData)
        vectPredict = np.float64(vectPredict) #For CosineSimilarity
        
        #The level of Compositionality :
        n1Similarity, n2Similarity = getLvlOfCompositionality(wordTest[i],i)
        
        listOfTuple = [] 
        for j in range(len(y_test)): #Look at all N1-de-N2
            vectToTest = np.zeros(shape=(1,300))
            vectToTest[0] = y_test[j]
            accuracy = tf.keras.losses.cosine_similarity(vectToTest,vectPredict)
            accuracy = accuracy.numpy()[0]
            wordTested = wordTest[j]
            tupleToAdd = (accuracy,wordTested)
            tupleGotInsered = False
            for k  in range(len(listOfTuple)): # Insert them in the right place in the list
                if(accuracy < listOfTuple[k][0]): #Find somewhere to insert in the right order 
                    listOfTuple.insert(k, tupleToAdd)
                    tupleGotInsered = True
                    break
            if(not tupleGotInsered):
                listOfTuple.append(tupleToAdd)
        listOfRank.append((getRank(wordTest[i], listOfTuple),wordTest[i],n1Similarity,n2Similarity))
        f = open("./fileProcess/Experience_2/"+str(listOfRank[-1][0])+"_"+wordTest[i], "w")
        f.write("Niveau de compositionalité N1 :\t" + str(n1Similarity)+ "\tN2 :\t" + str(n2Similarity) + "\n")
        for line in listOfTuple:
            f.write(str(line[0])+"\t"+line[1])
            if(line[1]==wordTest[i]):
                f.write("\tRank : " + str(listOfRank[-1][0]))
            f.write("\n")
        f.close()
    return listOfRank
            
    
def getRank(word,listOfTuple):
    for i in range(len(listOfTuple)):
        if(listOfTuple[i][1] == word):
            return i
    return -1


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

def getN1andN2(word):
    n1 = ""
    prep = ""
    n2 = ""
    tmp = ""
    nbrDotSeen = 0
    for letter in word:
        if(letter == "-"):
    
            nbrDotSeen+=1
            if(nbrDotSeen == 1):
                n1 = tmp
                tmp = ""
            elif(nbrDotSeen == 2):
                prep =  tmp
                tmp = ""
                
        else:
            tmp = tmp+letter
    n2 = tmp
    return n1,prep,n2

def selectRandomVector(setOfVect,nbr):
    vectorSelect = np.zeros(shape=(nbr,300))
    for i in range(nbr):
        randomNumber = rd.randint(0,len(setOfVect)-1)
        vectorSelect[i] = setOfVect[randomNumber]
    return vectorSelect

def evaluateResult(results):
    nbrOfGoodPrediction = 0
    for i in results:
        if(i == 1):
            nbrOfGoodPrediction+=1
    print("Il y a " +  str(len(results)) + " predictions")
    print(str(nbrOfGoodPrediction) + " sont de bonnes prédictions soit : " + str(nbrOfGoodPrediction*100/len(results)) + "%" )


def meanOfRank(rankList):
    sumRank = 0
    for rank in rankList:
        sumRank += rank[0]
    sumRank = sumRank/len(rankList)
    print("The Mean of all the rank is : " + str(sumRank))

def simpleMeanReciprocalRank(rankList):
    add = 0
    for nbr in rankList:
        add += 1/(nbr[0]+1)
        
    add = add*(1/len(rankList))
    
    print("Mean reciprocal rank : " + str(add))
    
def acteClassMRR(rankList):
    clActe = 0
    nbrActe = 0
    clOther = 0
    nbrOther = 0
    for tpl in rankList:
        if(tpl[1][0] == "a"):#Find the acte class
            clActe += 1/(tpl[0]+1)
            nbrActe+=1
        else:
            clOther += 1/(tpl[0]+1)
            nbrOther+=1
    mrrActe = clActe*(1/nbrActe)
    mrrOther = clOther*(1/nbrOther)
    print("MRR of N1=acte : " + str(mrrActe))
    print("MRR of other N1 : " + str(mrrOther))
        
def N1ClassMRR(rankList):
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
    mrrActe = clN1*(1/nbrN1)
    mrrOther = clN2*(1/nbrN2)
    print("MRR of N1<N2 : " + str(mrrActe))
    print("MRR of N2<N1 : " + str(mrrOther))



def sortByN1AndN2(rankList):
    rankList.sort(key = lambda x : x[2]+x[3])
    listRank = []
    for val in rankList:
        listRank.append(val[0])
    print(listRank)
    
    
rankList = evalModel_2(X_test,y_test,wordTest)

meanOfRank(rankList)
simpleMeanReciprocalRank(rankList)
acteClassMRR(rankList)
N1ClassMRR(rankList)
sortByN1AndN2(rankList)