#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:03:12 2021

@author: yannis.coutouly
"""

"""
This program is for preprocess the corpus for Word2Vec
"""

from gensim.models import Word2Vec

import os


prepTaken = ["de"]

mapPrepAndFollow = dict([("de", [])])

def processForWord2Vec():
    path = "./FRWAC"
    nbrCorpusProcess = 0
    for file in os.listdir(path):
        current = os.path.join(path, file)
        f = open("./FRWACProcess/" + str(nbrCorpusProcess) + ".txt" , "w")
        if os.path.isfile(current):
            data = open(current, "rb")
            for line in data.readlines():
                line = line.decode("utf-8")
                word = line.split("\t")[3].lower()
                if(word == "."):
                    f.write(word + "\n")
                else:
                    f.write(word+" ")
            data.close()
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
        f.close()

def processForWord2VecTogether():
    path = "../Corpora/conllCorpusProcess/ToRead"
    rawText= []
    tabText = []
    gramaticalType = []
    nbrCorpusProcess = 0
    
    for file in os.listdir(path):
        current = os.path.join(path, file)
        f = open("../Corpora/conllCorpusProcess/ToRead/" + str(nbrCorpusProcess) + ".txt" , "w")
        if os.path.isfile(current):
            data = open(current, "rb")
            nom1 = ""
            prep = ""
            prep2 = ""
            nom2 = ""
            for line in data.readlines():
                line = line.decode("utf-8")
                if(line[0] == "#"):
                    continue
                elif(line[0] == "\n"):
                    f.write("\n")
                    continue
                if(not(line.split("\t")[0].isnumeric())): #Get rid of he 1-2 line
                    continue
                word = line.split("\t")[1].lower()
                gramarWord = line.split("\t")[3]
                #tabText.append(word) pour gagner de la place en mémoire
                #gramaticalType.append(gramarWord) Pour gagner encore plus de place en mémoire
                if(nom1 == "" and gramarWord == "NOUN"): #findN1
                    nom1 = word
                    continue
                elif (nom1 != "" and  prep == ""): #find prep
                    if(gramarWord == "ADP"):
                        if(word in prepTaken):
                            prep = word
                            continue
                        else:
                            f.write(nom1 + " " + word + " ")
                            nom1
                    else:
                        f.write(nom1 + " " + word + " ")
                        nom1 = ""
                elif(prep != ""):
                    if(gramarWord == "NOUN"): #Is N1deN2
                        nom2 = word
                        f.write(nom1 + "-" + prep + prep2 + "-" + nom2 + " ")
                        nom1 = ""
                        prep = "" 
                        prep2 = ""
                        nom2 = ""
                    #elif(gramarWord == "DET"):
                        #prep2 = word 
                     #   continue
                    else:
                        f.write(nom1 + " " + prep + " " + prep2 + " " + word + " ")
                        nom1 = ""
                        prep = ""
                        prep2 = ""
                        nom2 = ""
                else:
                    nom1 = ""
                    prep = ""
                    prep2 = ""
                    nom2 = ""
                    f.write(word + " ")
                #if(word == "."):
                    #f.write("\n")
            nbrCorpusProcess+=1
            print(str(nbrCorpusProcess) + " Corpus traité")
            data.close()
        f.close()
    return rawText, gramaticalType, tabText
    
processForWord2VecTogether()