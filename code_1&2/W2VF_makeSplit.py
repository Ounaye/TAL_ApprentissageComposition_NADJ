#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:05:26 2021

@author: yannis.coutouly
"""
import heapq
#from itertools import izip
import numpy as np

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

class Embeddings:
   def __init__(self, vecsfile, vocabfile=None, normalize=True):
      if vocabfile is None: vocabfile = vecsfile.replace("npy","vocab")
      self._vecs = np.load(vecsfile)
      self._vocab = open(vocabfile).read().split()
      if normalize:
         self._vecs = ugly_normalize(self._vecs)
      self._w2v = {w:i for i,w in enumerate(self._vocab)}

   @classmethod
   def load(cls, vecsfile, vocabfile=None):
      return Embeddings(vecsfile, vocabfile)

   def word2vec(self, w):
      return self._vecs[self._w2v[w]]

   def similar_to_vec(self, v, N=10):
      sims = self._vecs.dot(v)
      sims = heapq.nlargest(N, zip(sims,self._vocab,self._vecs))
      return sims

   def most_similar(self, word, N=10):
      w = self._vocab.index(word)
      sims = self._vecs.dot(self._vecs[w])
      sims = heapq.nlargest(N, zip(sims,self._vocab))
      return sims

"""
   def analogy(self, pos1, neg1, pos2,N=10,mult=True):
      wvecs, vocab = self._vecs, self._vocab
      p1 = vocab.index(pos1)
      p2 = vocab.index(pos2)
      n1 = vocab.index(neg1)
      if mult:
         p1,p2,n1 = [(1+wvecs.dot(wvecs[i]))/2 for i in (p1,p2,n1)]
         if N == 1:
            return max(((v,w) for v,w in izip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
         return heapq.nlargest(N,((v,w) for v,w in izip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
      else:
         p1,p2,n1 = [(wvecs.dot(wvecs[i])) for i in (p1,p2,n1)]
         if N == 1:
            return max(((v,w) for v,w in izip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))
         return heapq.nlargest(N,((v,w) for v,w in izip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))
"""
###############################################################################

generalDataPath = "../BatchExp/SecondBatchExp/DataExperience/"
trainDataPath = generalDataPath+"Train"
testDataPath = generalDataPath+"Test"
devDataPath = generalDataPath+"Dev"
utilisDataPath = generalDataPath+"Utils"

e = Embeddings.load(generalDataPath + "Embeddings/vecs.npy")

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
###############################################################################
  
def makedictOfN1andN2():
    data = open(utilisDataPath+"/Conll7_AllFixedFormSorted.txt")
    for line in data.readlines():
        n1 = line.split("-")[0]
        for index in range(len(n1)):
            if(n1[index] == " " or n1[index].isdigit()):
                continue
            else:
                n1 = n1[index:]#Get only the word not the number of occurence
                break
        prep = line.split("-")[1]
        n2 = line.split("-")[2]
        n2 = n2[:len(n2)-1] #Remove the space at the end
        if(n1 in allN1):    
            mapOfN1andN2[n1].append((prep,n2))
  
###############################################################################
 
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

###############################################################################

def separateDataRandom():
    f = open(utilisDataPath + "/N1300_N2300Sorted.txt")
    testList = []
    trainList = []
    devList = []
    pseudoRandomCount = 0 
    for line in f.readlines():
        word = line
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

###############################################################################


def writeData(path,data):
    f = open(path +".txt","w")
    tabNumpy_X = np.zeros(shape=(len(data),600))
    tabNumpy_y = np.zeros(shape=(len(data),300))
    index = 0
    nbrProblem = 0
    for line in data:
        wordTog = line[0] + "-" + line[1] +"-" + line[2]
        if(not(wordTog in e._w2v)):
            print(wordTog)
            nbrProblem += 1
            continue
        vectTog = e.word2vec(wordTog)
        tabNumpy_y[index] = vectTog
        f.write(wordTog + "\n")
        wordN1 = line[0]
        wordN2 = line[2]
        if(not(wordN2 in e._w2v) or not(wordN1 in e._w2v)):
            print(wordTog)
            nbrProblem += 1
            continue
        vectN1 = e.word2vec(wordN1)
        vectN2 = e.word2vec(wordN2)
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

###############################################################################

def writeDataOnRandom(path,data):
    f = open(path +".txt","w")
    tabNumpy_X = np.zeros(shape=(len(data)-1998,600))
    tabNumpy_y = np.zeros(shape=(len(data)-1998,300))
    index = 0
    nbrProblem = 0
    for line in data:
        wordTog = line[:len(line)-1]
        if(not(wordTog in e._w2v)):
            #print(wordTog)
            nbrProblem += 1
            continue
        vectTog = e.word2vec(wordTog)
        tabNumpy_y[index] = vectTog
        wordN1 = wordTog.split("-")[0]
        wordN2 = wordTog.split("-")[2]
        if(not(wordN1 in e._w2v) or not(wordN2 in e._w2v)):
            print(wordTog + "part")
            nbrProblem += 1
            continue
        vectN1 = e.word2vec(wordN1)
        vectN2 = e.word2vec(wordN2)
        f.write(wordTog + "\n")
        vectN1N2 = np.hstack([vectN1,vectN2]) # Make the input vector
        tabNumpy_X[index] = vectN1N2
        index+=1
    f.close()
    print("On a trouvé : " + str(nbrProblem) + " probleme sur " + str(len(tabNumpy_X)))
    """
    for i in range(nbrProblem):
        if(i % 100 == 0):
            print(i)
        tabNumpy_X = np.delete(tabNumpy_X,len(data)-i-1,0)
        tabNumpy_y = np.delete(tabNumpy_y,len(data)-i-1,0)
    """
    np.save(path + "_X.npy",tabNumpy_X)
    np.save(path + "_y.npy",tabNumpy_y)
    
###############################################################################

