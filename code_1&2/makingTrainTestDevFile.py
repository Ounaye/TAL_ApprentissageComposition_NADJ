#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:26:04 2021

@author: yannis.coutouly
"""
"""
This file is for making and saving the Train, Test and Dev file

For Batch1 We need : Each file is one txt with the word and another one with the number in the main
embeddings
    
Pour les jeu d’entrainements :
Absolument tout sauf 10 % de chaque catégorie sauf 2 N1 train_N1=11but10
Le N=8 – 10 % 		train_N1=8but10

Pour les jeu test : 

10 % des N1=8  	test_N1=8but10
10 % des N1=11 	test_N1=11but10
10 % des N1=8 + 2N1	test_N1=8but10_N1=5
10 % des N1=11 + 2 N1	test_N1=11but10_N1=2


Pour faire le dev : On coupe nos test en deux et basta 

0 % des N1=8  	dev_N1=8but10
10 % des N1=11 	dev_N1=11but10
10 % des N1=8 + 2N1	dev_N1=8but10_N1=5
10 % des N1=11 + 2 N1	dev_N1=8but10_N1=2
"""

from gensim.models import Word2Vec
import numpy as np

generalDataPath = "../SecondBatchExp/DataExperience/"
trainDataPath = generalDataPath+"Train"
testDataPath = generalDataPath+"Test"
devDataPath = generalDataPath+"Dev"
embeddingsDataPath = generalDataPath+"Embeddings"
utilisDataPath = generalDataPath+"Utils"

# Il faut faire le train et le test en même temps
# Ecrire le train et diviser le test puis l'écrire

allN1 =  ["salle", "outil","jeu","règle","équipe","chef","liste","acte","zone","technique","question",
                "méthode", "droit"]
allN1Except2 = ["salle", "outil","jeu","règle","équipe","chef","liste","zone","question",
                "méthode", "droit"]

allN1Except5 = ["salle", "outil","jeu","règle","équipe","chef","liste","zone"]

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




def makedictOfN1andN2():
    data = open(utilisDataPath+"/sortedFilePrepCut_2.txt")
    for line in data.readlines():
        n1 = line.split("\t")[0]
        for index in range(len(n1)):
            if(n1[index] == " " or n1[index].isdigit()):
                continue
            else:
                n1 = n1[index:]#Get only the word not the number of occurence
                break
        prep = line.split("\t")[1]
        n2 = line.split("\t")[2]
        n2 = n2[:len(n2)-1] #Remove the space at the end
        if(n1 in allN1):    
            mapOfN1andN2[n1].append((prep,n2))
            

"""Train:
Absolument tout sauf 10 % de chaque catégorie sauf 2 N1 train_N1=11but10

Test :
10 % des N1=11 	test_N1=11but10
10 % des N1=11 + 2 N1	test_N1=8but10_N1=2

Dev:
10 % des N1=11 	dev_N1=11but10
10 % des N1=11 + 2 N1	dev_N1=8but10_N1=2
"""
def separateDataN11(includeN1):
    pseudoRandom = 0
    pseudoRandom_2 = 0
    testList = []
    trainList = []
    devList = []
    for n1 in allN1Except2:
        n2List = mapOfN1andN2.get(n1)
        for n2Tuple in n2List:
            prep = n2Tuple[0]
            n2 = n2Tuple[1]
            if(pseudoRandom % 10 == 0): #Go in Test (10%)
                if(pseudoRandom_2 % 2 == 0): # Go in Test (50%)
                    testList.append((n1,prep,n2))
                else: # Go in dev (50%)
                    devList.append((n1,prep,n2))
                pseudoRandom +=1
                pseudoRandom_2 +=1 
            else: # Go in Train
                trainList.append((n1,prep,n2))
                pseudoRandom +=1
    if(includeN1):
        for n1 in allN1:
            if(not (n1 in allN1Except2)): # Is in the 2 N1 
                n2List = mapOfN1andN2.get(n1)
                for n2Tuple in n2List:
                    prep = n2Tuple[0]
                    n2 = n2Tuple[1]
                    if(pseudoRandom_2 % 2 == 0): # Go in Test (50%)
                        testList.append((n1,prep,n2))
                    else: # Go in dev (50%)
                        devList.append((n1,prep,n2))
                    pseudoRandom_2 +=1 
    return trainList,testList,devList
"""

Le N=8 – 10 % 		train_N1=8but10

Pour les jeu test : 

10 % des N1=8  	test_N1=8but10
10 % des N1=8 + 5N1	test_N1=8but10_N1=5


Pour faire le dev : On coupe nos test en deux et basta 

10 % des N1=8  	dev_N1=8but10
10 % des N1=8 + 5N1	dev_N1=8but10_N1=5
"""

def separateDataN8(includeN1):
    pseudoRandom = 0
    pseudoRandom_2 = 0
    testList = []
    trainList = []
    devList = []
    for n1 in allN1Except5:
        n2List = mapOfN1andN2.get(n1)
        for n2Tuple in n2List:
            prep = n2Tuple[0]
            n2 = n2Tuple[1]
            if(pseudoRandom % 10 == 0): #Go in Test (10%)
                if(pseudoRandom_2 % 2 == 0): # Go in Test (50%)
                    testList.append((n1,prep,n2))
                else: # Go in dev (50%)
                    devList.append((n1,prep,n2))
                pseudoRandom +=1
                pseudoRandom_2 +=1 
            else: # Go in Train
                trainList.append((n1,prep,n2))
                pseudoRandom +=1
    if(includeN1):
        for n1 in allN1:
            if(not (n1 in allN1Except5)): # Is in the 5 N1 
                n2List = mapOfN1andN2.get(n1)
                for n2Tuple in n2List:
                    prep = n2Tuple[0]
                    n2 = n2Tuple[1]
                    if(pseudoRandom_2 % 2 == 0): # Go in Test (50%)
                        testList.append((n1,prep,n2))
                    else: # Go in dev (50%)
                        devList.append((n1,prep,n2))
                    pseudoRandom_2 +=1 
    return trainList,testList,devList
        
"""
Separate data in 70/15/15  Train/Test/Dev
"""




def separateDataRandom():
    f = open(utilisDataPath + "/best300Full10N1N2.txt")
    testList = []
    trainList = []
    devList = []
    pseudoRandomCount = 0 
    for line in f.readlines():
        word = line
        lastWord = False
        indexLastWord = 0
        """
        for i in range(len(line)):
            if(not(lastWord) and not(line[i] == " ")):
                lastWord = True
                indexLastWord = i
                continue
            if(lastWord and line[i] == "-"):
                word = line[indexLastWord:]
                break
            if(lastWord and line[i] == " "):
                lastWord = False"""
        if(pseudoRandomCount < 14):
            trainList.append(word)
        elif(pseudoRandomCount < 17):
            testList.append(word)
        else:
            devList.append(word)
        pseudoRandomCount +=1
        if(pseudoRandomCount == 20):
            pseudoRandomCount = 0
    return trainList,testList,devList
    

# The split is done now write the text and save the npTab



def writeData(path,data):
    f = open(path +".txt","w")
    tabNumpy_X = np.zeros(shape=(len(data),600))
    tabNumpy_y = np.zeros(shape=(len(data),300))
    index = 0
    nbrProblem = 0
    for line in data:
        wordTog = line[0] + "-" + line[1] +"-" + line[2]
        if(not(wordTog in embeddingsTog.wv)):
            print(wordTog)
            nbrProblem += 1
            continue
        vectTog = embeddingsTog.wv[wordTog]
        tabNumpy_y[index] = vectTog
        f.write(wordTog + "\n")
        wordN1 = line[0]
        wordN2 = line[2]
        vectN1 = embeddingsTog.wv[wordN1]
        vectN2 = embeddingsTog.wv[wordN2]
        vectN1N2 = np.hstack([vectN1,vectN2]) # Make the input vector
        tabNumpy_X[index] = vectN1N2
        index+=1
    f.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(tabNumpy_X)))
    for i in range(nbrProblem):
        tabNumpy_X = np.delete(tabNumpy_X,len(data)-i-1,0)
        tabNumpy_y = np.delete(tabNumpy_y,len(data)-i-1,0)
    
    np.save(path + "_X.npy",tabNumpy_X)
    np.save(path + "_y.npy",tabNumpy_y)


def writeDataOnRandom(path,data):
    f = open(path +".txt","w")
    tabNumpy_X = np.zeros(shape=(len(data),600))
    tabNumpy_y = np.zeros(shape=(len(data),300))
    index = 0
    nbrProblem = 0
    for line in data:
        wordTog = line[:len(line)-1]
        if(not(wordTog in embeddingsTog.wv)):
            print(wordTog)
            nbrProblem += 1
            continue
        vectTog = embeddingsTog.wv[wordTog]
        tabNumpy_y[index] = vectTog
        f.write(wordTog + "\n")
        wordN1 = wordTog.split("-")[0]
        wordN2 = wordTog.split("-")[2]
        if(not(wordN1 in embeddingsTog.wv) or not(wordN2 in embeddingsTog.wv)):
            print(wordTog + "part")
            nbrProblem += 1
            continue
        vectN1 = embeddingsTog.wv[wordN1]
        vectN2 = embeddingsTog.wv[wordN2]
        vectN1N2 = np.hstack([vectN1,vectN2]) # Make the input vector
        tabNumpy_X[index] = vectN1N2
        index+=1
    f.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(tabNumpy_X)))
    for i in range(nbrProblem):
        if(i % 100 == 0):
            print(i)
        tabNumpy_X = np.delete(tabNumpy_X,len(data)-i-1,0)
        tabNumpy_y = np.delete(tabNumpy_y,len(data)-i-1,0)
    
    np.save(path + "_X.npy",tabNumpy_X)
    np.save(path + "_y.npy",tabNumpy_y)

print("Load the Embeddings")
makedictOfN1andN2()
#embeddings = Word2Vec.load(embeddingsDataPath+"/embeddings_300_Prep=4_ N1=13.model")
embeddingsTog = Word2Vec.load(embeddingsDataPath+"/embeddings_300_full10Fusion_V2.model")



trainList,testList,devList = separateDataN8(True)


print("When we write some N1-de-N2 display, they will be exclude of the corpus because of some strange bug")
writeData(trainDataPath+"/train_N1=11but10",trainList)
writeData(testDataPath+"/test_N1=8but10_N1=5",testList)
writeData(devDataPath+"/dev_N1=8but10_N1=5", devList)














