#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:00:17 2021

@author: yannis.coutouly

Neural Network V2 : 
    
We will use the the model of NN propose by alexis with additionnal information for the decoding part
Moreover we will give in input 2 N_ADJ couple The NN will be the same at this point
the only things that change is the loss calculation, it has to be a custom loss.
Let's take : N1_A1 as the first couple and N2_A2 as the second couple. The loss will be :
    
    cos(N1,N1') + cos(A1,A1') + 1 – abs([1,-1] * [cos(N1_A1,N2_A2) , cos(N1,N2) + cos(A1,A2)])
    
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import testModel as tM

generalDataPath = "../BatchExp/ForthBatchExp/DataExperience/"
trainDataPath = generalDataPath+"Train"
devDataPath = generalDataPath+"Dev"
neuralNetworkPath = generalDataPath + "NeuralNetwork"


typeTrain = "/train_N_ADJ_Full"
typeOfDev = "/dev_N_ADJ_Full"

#Load Data --------------------------------------------------------------------------------------

X_train = np.load(trainDataPath + typeTrain + ".npy")
X_dev = np.load(devDataPath + typeOfDev + ".npy")


arrDev = np.split(X_dev,[300],axis=1)
word1_Dev = arrDev[0]
word2_Dev = arrDev[1]
arrTrain = np.split(X_train,[300],axis=1)
word1_Train = arrTrain[0]
word2_Train = arrTrain[1]

X_trainShuffle = np.copy(X_train)
np.random.shuffle(X_trainShuffle)
X_DevShuffle = np.copy(X_dev)
np.random.shuffle(X_DevShuffle)

arrDev2 = np.split(X_DevShuffle,[300],axis=1)
word1_Dev2 = arrDev2[0]
word2_Dev2 = arrDev2[1]
arrTrain2 = np.split(X_trainShuffle,[300],axis=1)
word1_Train2 = arrTrain2[0]
word2_Train2 = arrTrain2[1]

print("Data is loaded")
#Convert To tensor

X_train = tf.convert_to_tensor(X_train)
X_dev = tf.convert_to_tensor(X_dev)

word1_Dev = tf.convert_to_tensor(word1_Dev)
word2_Dev = tf.convert_to_tensor(word2_Dev)
word1_Train = tf.convert_to_tensor(word1_Train)
word2_Train = tf.convert_to_tensor(word2_Train)

X_train2 = tf.convert_to_tensor(X_trainShuffle)
X_dev2 = tf.convert_to_tensor(X_DevShuffle)

word1_Dev2 = tf.convert_to_tensor(word1_Dev2)
word2_Dev2 = tf.convert_to_tensor(word2_Dev2)
word1_Train2 = tf.convert_to_tensor(word1_Train2)
word2_Train2 = tf.convert_to_tensor(word2_Train2)

word1_Train = tf.cast(word1_Train, tf.float64)
word2_Train = tf.cast(word2_Train, tf.float64)
word1_Train2 = tf.cast(word1_Train2, tf.float64)
word2_Train2 = tf.cast(word2_Train2, tf.float64)

#Construct the model------------------------------------------------------------------------------

        # The first part of the NN
inputs = keras.Input(shape=(600,),name="nadj",dtype=tf.float64)
inputsAdj = keras.Input(shape=(300,),name="inAdj",dtype=tf.float64)
inputsNoun = keras.Input(shape=(300,),name="inNoun",dtype=tf.float64)
x = layers.Dense(300, activation="relu",name="N_ADJSpace",dtype=tf.float64)(inputs)

mergedAdj = keras.layers.Concatenate(axis=1)([x, inputsAdj])
mergedNoun = keras.layers.Concatenate(axis=1)([x, inputsNoun])

outputs_Noun = layers.Dense(300,name="nounDecoder")(mergedAdj)
outputs_Adj = layers.Dense(300,name="adjDecoder")(mergedNoun)

    # The second part of the NN

inputs2 = keras.Input(shape=(600,),name="nadj2",dtype=tf.float64)
inputsAdj2 = keras.Input(shape=(300,),name="inAdj2",dtype=tf.float64)
inputsNoun2 = keras.Input(shape=(300,),name="inNoun2",dtype=tf.float64)
x2 = layers.Dense(300, activation="relu",name="N_ADJSpace2")(inputs2)

mergedAdj2 = keras.layers.Concatenate(axis=1)([x2, inputsAdj2])
mergedNoun2 = keras.layers.Concatenate(axis=1)([x2, inputsNoun2])

outputs_Noun2 = layers.Dense(300,name="nounDecoder2")(mergedAdj2)
outputs_Adj2 = layers.Dense(300,name="adjDecoder2")(mergedNoun2)

model = tM.testModel(inputs=[inputs,inputsAdj,inputsNoun,inputs2,inputsAdj2,inputsNoun2], 
                    outputs=[outputs_Noun, outputs_Adj, outputs_Adj2,outputs_Noun2,x,x2], name="encoderDecoder")


model.summary()

# Set-up the training session------------------------------------------------------------------



def lossCalculation(similarityDecode,nSimilarity,aSimilarity):
    a = similarityDecode # cos(N1A1,N2A2)
    b = 0.8 * nSimilarity +  0.2 * aSimilarity # cos(N1,N2) + cos(A1,A2)
    
    constant = tf.constant([1,-1],dtype=tf.float64)
    value = tf.stack([a,b],)
    mappedTensor = tf.multiply(constant, value)
    return 1-tf.norm(mappedTensor,ord=1) # Easiest way to get what we want 

def my_lossTest(y_pred,N1,A1,N2,A2): #A tab of min batch

    nAdjSpace1 = y_pred[4]
    nAdjSpace2 = y_pred[5]
    
    nSimilarity = keras.losses.cosine_similarity(N1,N2)
    aSimilarity = keras.losses.cosine_similarity(A1,A2)
    similarityN1 = keras.losses.cosine_similarity(N1,tf.cast(y_pred[0], tf.float64))
    similarityA1 = keras.losses.cosine_similarity(A1,tf.cast(y_pred[1], tf.float64))
    similarityN2 = keras.losses.cosine_similarity(N2,tf.cast(y_pred[2], tf.float64))
    similarityA2 = keras.losses.cosine_similarity(A2,tf.cast(y_pred[3], tf.float64))
    similarityNAdj = keras.losses.cosine_similarity(tf.cast(nAdjSpace2, tf.float64),tf.cast(nAdjSpace1, tf.float64))
    

    
    similarity = tf.map_fn(fn= lambda x: lossCalculation(x[0],x[1],x[2]),
                           elems=(nSimilarity,aSimilarity,similarityNAdj),dtype=tf.float64)


    return (0.4*similarityN1 + 0.1*similarityA1 + 0.4*similarityN2 + 0.1*similarityA2) + similarity


model.compile(optimizer="adam",
    my_loss=my_lossTest)


my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3,monitor="loss",min_delta=0.001)]
print("Fit data")

model.fit({"nadj" : X_train, "inNoun" : word1_Train, "inAdj" : word2_Train, "nadj2" : X_train2, 
           "inNoun2" : word1_Train2, "inAdj2" : word2_Train2},
          {"nounDecoder": word1_Train, "adjDecoder": word2_Train, 
           "nounDecoder2" : word1_Train2, "adjDecoder2" : word2_Train2,
           "N_ADJSpace" : word1_Train, "N_ADJSpace2" : word2_Train2}   # The two last should be ignored
          ,epochs=50,validation_data=(
              {"nadj" : X_dev, "inNoun" : word1_Dev, "inAdj" : word2_Dev, 
               "nadj2" : X_dev2, "inNoun2" : word1_Dev2, "inAdj2" : word2_Dev2},
              {"nounDecoder": word1_Dev, "adjDecoder": word2_Dev, 
               "nounDecoder2" : word1_Dev2, "adjDecoder2" : word2_Dev2
               }), callbacks=my_callbacks)


#Make Copy of the Model --------------------------------------------------------------------


modelEncodeur = keras.Model(inputs=inputs, outputs=x, name="encoder")
modelEncodeur.summary()
modelEncodeur.compile()
modelEncodeur.save(neuralNetworkPath + "/Encode_N_ADJ_V2_lossV2Adjusted")


model.save(neuralNetworkPath + "/NN_N_ADJ_V2_lossV2Adjusted")