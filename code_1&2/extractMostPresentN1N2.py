#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:39:07 2021

@author: yannis.coutouly
"""

"""
A file to process some text file because i don't know how to use awk


"""
import os
import math

generalDataPath = "../BatchExp/SecondBatchExp/DataExperience/"
utilisDataPath = generalDataPath+"Utils"

data = open("../BatchExp/SecondBatchExp/DataExperience/Utils/conll7_300N1_N2Sorted300.txt")

dataN1 = open(utilisDataPath + "/coll7N1Sorted300.txt")
dataN2 = open(utilisDataPath + "/conll7_300N1_N2Sorted300.txt")
dataN1N2 = open(utilisDataPath + "/Conll7_AllFixedForm.txt")
dataWrite = open(utilisDataPath + "/compatible.txt","w")


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
def getAllN1N2(): # data is the best 300 N1 and we write only the N1-de-N2 where N1 is a 300 best
    f = open(utilisDataPath + "/conll7_300N1FixedForm.txt","w")
    index = 0
    for line in data:
         word = getWordInLine(line)
         #n1Word = word.split("-")[0]
         word = word[:len(word)-1]
         dataFull = open(utilisDataPath + "/Conll7_AllFixedForm.txt")
         if(index % 20 == 0):
             print(index)
         index += 1
         for lineFull in dataFull:
             wordFull = getWordInLine(lineFull)
             tmp = wordFull.split("-")[0]
             wordFull = wordFull[:len(wordFull)-1]
             if(tmp == word):
                 f.write(wordFull + "\n")
         dataFull.close()
    f.close()
def getCompatibility():
    n1Tab = []
    n2Tab = []
    for n1Line in dataN1:
        n1Word = getWordInLine(n1Line)
        n1Word = n1Word[:len(n1Word)-1]
        n1Tab.append(n1Word)
    for n2Line in dataN2:
        n2Word = getWordInLine(n2Line)
        n2Word = n2Word[:len(n2Word)-1]
        n2Tab.append(n2Word)
    for n1n2Line in dataN1N2:
        n1n2Word = n1n2Line[8:]
        n1n2Word = getWordInLine(n1n2Line)
        n1n2Word = n1n2Word[:len(n1n2Word)-1]
        n1WordFull = n1n2Word.split("-")[0]
        n2WordFull = n1n2Word.split("-")[2]
        for n1Word in n1Tab:
            if(n1Word == n1WordFull):
                for n2Word in n2Tab:
                    if(n2Word == n2WordFull):
                        dataWrite.write(n1n2Word + "\n")
    
        
"""
On va essayer de regarder s'il y a beaucoup de N1 qui s'enlève dans le together
On récupère tout nos noms dans la liste des 300 les plus vus
On parcours FRWAC/Together20 a chaque espace ou tiret on regarde si le mot avant est dans nos 300
Si c'est le cas alors on l'ajoute à un de nos conteurs

"""        
def countN1AloneAgainstTogether():
    path = "./FRWACProcess/N1N2Together20"
    dictN1Occurence = dict()
    listOfN1 = []
    nbrCorpusProcess = 0
    
    nom1File = open(utilisDataPath + "/mostPresentN1Cut300.txt","r")
    for line in nom1File:
        wordN1 = getWordInLine(line)
        wordN1 = wordN1[:len(wordN1)-1]
        dictN1Occurence.setdefault(wordN1,(0,0))
        listOfN1.append(wordN1) 
        listOfN1.sort()
    findByDichotomie(0,len(listOfN1),"note",listOfN1)
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            data = open(current, "rb")
            for line in data:
                line = line.decode("utf-8")
                for word in line.split(" "):
                    if("-" in word): #Maybie a N1-de-N2
                        for littleWord in word.split("-"):
                            index = findByDichotomie(0,len(listOfN1),littleWord,listOfN1)
                            if(listOfN1[index] == littleWord):
                                dictN1Occurence[littleWord] = addTuple2(dictN1Occurence.get(littleWord), (0,1))
                    else:
                        index = findByDichotomie(0,len(listOfN1),word,listOfN1)
                        if(listOfN1[index] == word): #Is an N1 
                                dictN1Occurence[word] = addTuple2(dictN1Occurence.get(word), (1,0))      
            data.close()
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
    return dictN1Occurence,listOfN1
        

def addTuple2(x,y):
    return (x[0]+y[0],x[1]+y[1])
        
def findByDichotomie(indexStart,indexEnd,elt,listOfObject):
    if(indexEnd - indexStart < 2):
        if(listOfObject[indexStart] == elt):
            return indexStart
        else: 
            return -1
    diff = indexEnd - indexStart
    newIndex = indexStart +  diff/2
    newIndex = math.floor(newIndex)
    if(elt < listOfObject[newIndex]):
        return findByDichotomie(indexStart, newIndex, elt, listOfObject)
    else:
        return findByDichotomie(newIndex, indexEnd, elt, listOfObject)
             

def writeDictResultOnFile(dictN1,listN1):
    f = open(utilisDataPath + "/N1AloneVsN1Together.txt", "w")
    for n1 in listN1:
        value = dictN1[n1]
        f.write(n1 + " Alone:" + str(value[0]) + " Together: " + str(value[1]) + "\n")

def readResultAndMean():
    data = open(utilisDataPath + "/N1AloneVsN1Together.txt", "r")
    nbrAlone = 0
    nbrTogether = 0
    for line in data:
        index = 0
        for word in line.split(" "):
            if(word[0] == "A"):
                value = word[6:]
                nbrAlone += int(value)
            elif(index ==3):
                value = word
                nbrTogether += int(value)
            index +=1
    print("Nbr Alone : " + str(nbrAlone) + " nbr Together : " + str(nbrTogether))
    print("On a en moyenne : " + str(nbrTogether/(nbrTogether + nbrAlone)) + " de N1 qui sont consomés ")
        
        
#dictN1,listN1 = countN1AloneAgainstTogether()
#writeDictResultOnFile(dictN1,listN1)

#getAllN1N2()
#getCompatibility()

getCompatibility()