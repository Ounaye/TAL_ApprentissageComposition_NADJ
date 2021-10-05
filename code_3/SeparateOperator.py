#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:22:24 2021

@author: yannis.coutouly

This file is to construct the encoder and decoder 

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


generalDataPath = "../BatchExp/ThirdBatchExp/DataExperience/"
trainDataPath = generalDataPath+"Train"
devDataPath = generalDataPath+"Dev"
neuralNetworkPath = generalDataPath + "NeuralNetwork"


typeTrain = "/train_Full_X"
typeOfDev = "/dev_Full_X"

#Load Data --------------------------------------------------------------------------------------

X_train = np.load(trainDataPath + typeTrain + ".npy")
X_dev = np.load(devDataPath + typeOfDev + ".npy")

arrDev = np.split(X_dev,[300],axis=1)
word1_Dev = arrDev[0]
word2_Dev = arrDev[1]
arrTrain = np.split(X_train,[300],axis=1)
word1_Train = arrTrain[0]
word2_Train = arrTrain[1]

model = tf.keras.models.load_model(generalDataPath + "NeuralNetwork/Encode_Train=WACFull10_Encode=300")

print("Data is loaded")
#Convert To tensor

X_train = tf.convert_to_tensor(X_train)
X_dev = tf.convert_to_tensor(X_dev)

word1_Dev = tf.convert_to_tensor(word1_Dev)
word2_Dev = tf.convert_to_tensor(word2_Dev)
word1_Train = tf.convert_to_tensor(word1_Train)
word2_Train = tf.convert_to_tensor(word2_Train)

X_train = model.predict(X_train)
X_dev = model.predict(X_dev)

#Construct the model------------------------------------------------------------------------------

inputs = keras.Input(shape=(300,),name="noun_adjSpace")
outputs_Noun = layers.Dense(300,name="nounDecoder")(inputs)
outputs_Adj = layers.Dense(300,name="adjDecoder")(inputs)

modelAdj = keras.Model(inputs=inputs, outputs=outputs_Adj, name="DecoderAdj")
modelNom = keras.Model(inputs=inputs, outputs=outputs_Noun, name="DecoderNoun")

modelAdj.summary()
modelAdj.compile(optimizer="adam",loss=tf.keras.losses.cosine_similarity)

modelNom.summary()
modelNom.compile(optimizer="adam",loss=tf.keras.losses.cosine_similarity)

my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3)]

print("Fit data")

print("modelAdj Fit ")

modelAdj.fit([[X_train]],[[word2_Train]],epochs=300,validation_data=(X_dev,word2_Dev),callbacks=my_callbacks)

print("modelNoun Fit ")

modelNom.fit([[X_train]],[[word1_Train]],epochs=300,validation_data=(X_dev,word1_Dev),callbacks=my_callbacks)

modelAdj.save(neuralNetworkPath + "/DecodeAdj_Train=WACFull10_Encode=300")
modelNom.save(neuralNetworkPath + "/DecodeNoun_Train=WACFull10_Encode=300")