# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import gensim 

from gensim.models import Word2Vec

import os

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


prepTaken = ["de"]

mapPrepAndFollow = dict([("de", [])])



def makedictOfN1andN2(): 
    data = open("./fileProcess/MakingEmbeddings/sortedFilePrepCut_2.txt")
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
        if(isInTheN1Taken(n1)):    
            mapOfN1andN2[n1].append((prep,n2))
        
def isInTheN1Taken(word):
    for wordN1 in tabOfN1Taken:
        if(wordN1 == word):
            return True
    return False

def isInTheMap(n1,n2):
    for word in mapOfN1andN2.get(n1):
        if(word[1] == n2):
            return True
    return False

print("wait")



def extractTextInAllFileAndRegroupN1deN2(): 
    path = "./FRWAC"
    rawText= []
    tabText = []
    gramaticalType = []
    nbrCorpusProcess = 0
    
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            data = open(current, "rb")
            textLine = []
            nom1 = ""
            prep = ""
            prep2 = ""
            nom2 = ""
            isPresent = False
            for line in data.readlines():
                line = line.decode("utf-8")
                word = line.split("\t")[3].lower()
                gramarWord = line.split("\t")[1]
                #tabText.append(word) pour gagner de la place en mémoire
                #gramaticalType.append(gramarWord) Pour gagner encore plus de place en mémoire
                if(nom1 == "" and gramarWord == "NOUN" and isInTheN1Taken(word)): #findN1
                    nom1 = word
                    continue
                elif (nom1 != "" and  prep == ""): #find prep
                    if(word in prepTaken):
                        prep = word
                        continue
                    else:
                        textLine.append(nom1)
                        textLine.append(word)
                        nom1 = ""
                elif(prep != ""):
                    if(gramarWord == "NOUN" and isInTheMap(nom1, word)): #Is N1deN2
                        nom2 = word
                        textLine.append(nom1 + "-" + prep + prep2 + "-" + nom2)
                        #textLine.append(nom1)
                        #textLine.append(prep)
                        #textLine.append(prep2)
                        #textLine.append(nom2)
                        isPresent = True
                        nom1 = ""
                        prep = "" 
                        prep2 = ""
                        nom2 = ""
                    elif(gramarWord == "DET"):
                        prep2Word = mapPrepAndFollow.get(prep)
                        if(word in prep2Word):
                            prep2 = word 
                            continue
                    else:
                        textLine.append(nom1)
                        textLine.append(prep)
                        textLine.append(prep2)
                        textLine.append(word)
                        nom1 = ""
                        prep = ""
                        prep2 = ""
                        nom2 = ""
                else:
                    nom1 = ""
                    prep = ""
                    prep2 = ""
                    nom2 = ""
                    textLine.append(word)
                if(word == "."):
                    if(isPresent):
                        rawText.append(textLine)
                        isPresent = False
                    textLine = []
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
            print(len(rawText))
            data.close()
    return rawText, gramaticalType, tabText


def extractN1deN2InAllFileAndWrite(): # Extract N1-de-N2 and write in a text
    path = "../Corpora/conllCorpusProcess/ToRead"
    rawText= []
    gramaticalType = []
    f = open("fileOf_N1deN2.txt", "w")
    nbrCorpusProcess = 0
    
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            data = open(current, "rb")
            nom1 = ""
            prep = ""
            prep2 = ""
            nom2 = ""
            for line in data.readlines():
                lineText = line
                lineText = lineText.decode("utf-8")
                if(lineText[0] == "#" or lineText[0] == "\n"):
                    continue
                word =lineText.split("\t")[2].lower()
                gramar = lineText.split("\t")[3]
                if(nom1 == "" and gramar == "NOUN"): #findN1
                    nom1 = word
                    continue
                elif (nom1 != "" and  prep == ""): #find prep
                    if(word in prepTaken):
                        prep = word
                        continue
                    else:
                        nom1 = ""
                elif(prep != ""):
                    if(gramar == "NOUN"):
                        nom2 = word
                        f.write(nom1 + "\t" + prep + prep2 + "\t" + nom2 + "\n")
                        nom1 = ""
                        prep = ""
                        nom2 = ""
                    elif(gramar == "DET"):
                        prep2Word = mapPrepAndFollow.get(prep)
                        if(word in prep2Word):
                            prep2 = word 
                            continue
                else:
                    nom1 = ""
                    prep = ""
                    prep2 = ""
                    nom2 = ""
            data.close()
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
    f.close()
    return rawText, gramaticalType


def processBigDataOutOfTog(rawText):
    for line in rawText:
        indexWord = 0
        for word in line:
            findComposite = 0
            for letter in word:
                if(letter == "-"):
                    findComposite +=1
            if(findComposite == 2):
                n1Word = word.split("-")[0]
                prepWord = word.split("-")[1]
                n2Word = word.split("-")[2]
                line[indexWord] = n1Word
                line.insert(indexWord+1, prepWord)
                line.insert(indexWord+2,n2Word)
                findComposite = 0
            indexWord +=1
    return rawText
                
            
def extractN1deN2InProcessFile(): # Extract N1-de-N2 of FRWACProcess and write in a text
     path = "../Corpora/conllCorpusProcess/ToRead"
     f = open("fileOf_N1deN2.txt", "w")
     nbrCorpusProcess = 0
     for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isfile(current):
            data = open(current, "rb")
            for line in data.readlines():
                line = line.decode("utf-8")
                for word in line.split(" "):
                    if("-de-" in word or "-dele-" in word or "-du-" in word):
                        f.write(word + "\n")
            data.close()
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
     f.close()



extractN1deN2InProcessFile()