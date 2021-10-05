#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:16:19 2021

@author: yannis.coutouly

In this file i code the NN of encoder decoder 

I have to make my own loss and break my neural network in 2 part to get the two operation

https://keras.io/guides/functional_api/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

print("Data is loaded")
#Convert To tensor

X_train = tf.convert_to_tensor(X_train)
X_dev = tf.convert_to_tensor(X_dev)

word1_Dev = tf.convert_to_tensor(word1_Dev)
word2_Dev = tf.convert_to_tensor(word2_Dev)
word1_Train = tf.convert_to_tensor(word1_Train)
word2_Train = tf.convert_to_tensor(word2_Train)

#Construct the model------------------------------------------------------------------------------

inputs = keras.Input(shape=(600,),name="nadj")
inputsAdj = keras.Input(shape=(300,),name="inAdj")
inputsNoun = keras.Input(shape=(300,),name="inNoun")
x = layers.Dense(400, activation="relu",name="N_ADJSpace")(inputs)
mergedAdj = keras.layers.Concatenate(axis=1)([x, inputsAdj])
mergedNoun = keras.layers.Concatenate(axis=1)([x, inputsNoun])
outputs_Noun = layers.Dense(300,name="nounDecoder")(mergedAdj)
outputs_Adj = layers.Dense(300,name="adjDecoder")(mergedNoun)


model = keras.Model(inputs=[inputs,inputsAdj,inputsNoun], outputs=[outputs_Noun, outputs_Adj], name="encoderDecoder")

model.summary()

model.compile(optimizer="adam",
    loss={
        "nounDecoder": tf.keras.losses.cosine_similarity,
        "adjDecoder": tf.keras.losses.cosine_similarity},
    loss_weights={"nounDecoder": 0.9, "adjDecoder": 0.1})


my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3,monitor="nounDecoder_loss",min_delta=0.001)]
print("Fit data")

model.fit({"nadj" : X_train, "inNoun" : word1_Train, "inAdj" : word2_Train},
          {"nounDecoder": word1_Train, "adjDecoder": word2_Train}
          ,epochs=300,validation_data=({"nadj" : X_dev, "inNoun" : word1_Dev, "inAdj" : word2_Dev},
                                       {"nounDecoder": word1_Dev, "adjDecoder": word2_Dev}), callbacks=my_callbacks)


#Make Copy of the Model --------------------------------------------------------------------


modelEncodeur = keras.Model(inputs=inputs, outputs=x, name="encoder")
modelEncodeur.summary()
modelEncodeur.compile()
modelEncodeur.save(neuralNetworkPath + "/Encode_N_ADJ_400_Noun=9_V2")


model.save(neuralNetworkPath + "/NN_N_ADJ_400_Noun=9_V2")
