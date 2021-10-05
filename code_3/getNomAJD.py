# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

This file is to extract the information of N_ADJ in the FRWAC corpus
And DET_ADJ
and DET_NOUN_ADJ

"""


generalPath = "../BatchExp/ForthBatchExp/DataExperience/"
corpusPath = "../Corpora/FRWAC/0"

def getN_ADJAndWrite():
    writeFile = open(generalPath + "Utils/N_ADJ_Full10.txt","w")
    for corpusNumber in range(9):
        print("Corpus n°"+str(corpusNumber))
        f = open(corpusPath + str(corpusNumber+1)+".mcf")
        nom = ""
        adj = ""
        for line in f.readlines():
            gramar = line.split("\t")[1]
            lemme = line.split("\t")[3]
            lemme = str.lower(lemme)
            if(gramar == "ADJ"):
                adj = lemme
            elif(gramar == "NOUN"):
                nom = lemme
            else:
                nom = ""
                adj = ""
                continue
            if(nom != "" and adj != ""):
                writeFile.write(nom + "-" + adj + "\n")
        f.close()
            
        
def getN_DetAndWrite():
    writeFile = open(generalPath + "Utils/N_Det_Full10.txt","w")
    for corpusNumber in range(9):
        print("Corpus n°"+str(corpusNumber))
        f = open(corpusPath + str(corpusNumber+1)+".mcf")
        nom = ""
        det = ""
        for line in f.readlines():
            gramar = line.split("\t")[1]
            lemme = line.split("\t")[3]
            lemme = str.lower(lemme)
            if(gramar == "DET"):
                det = lemme
            elif(gramar == "NOUN"):
                nom = lemme
            else:
                nom = ""
                det = ""
                continue
            if(nom != "" and det != ""):
                writeFile.write(det + "-" + nom + "\n")
        f.close()


def getDet_N_AdjAndWrite():
    writeFile = open(generalPath + "Utils/Det_Noun_Adj_Full10.txt","w")
    for corpusNumber in range(9):
        print("Corpus n°"+str(corpusNumber))
        f = open(corpusPath + str(corpusNumber+1)+".mcf")
        nom = ""
        adj = ""
        det = ""
        for line in f.readlines():
            gramar = line.split("\t")[1]
            lemme = line.split("\t")[3]
            lemme = str.lower(lemme)
            if(gramar == "DET"):
                det = lemme
            elif(det != "" and gramar == "ADJ"):
                adj = lemme
            elif(det != "" and gramar == "NOUN"):
                nom = lemme
            else:
                det = ""
                adj = ""
                nom = ""
            if(nom != "" and adj != ""):
                writeFile.write(det + "-" + nom + "-" + adj +"\n")
                det = ""
                adj = ""
                nom = ""
        f.close()
    
getDet_N_AdjAndWrite()