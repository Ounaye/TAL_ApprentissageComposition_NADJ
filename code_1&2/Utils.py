#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:14 2021

@author: yannis.coutouly
"""

"""

Des fonctions qui restent utiles même si elles ne sont plus utilisé 

"""


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