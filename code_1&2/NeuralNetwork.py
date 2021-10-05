#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:22:43 2021

@author: yannis.coutouly
"""

"""
Ce programme a pour but de contenir notre réseaux de neurones
De récupérer les données dans ExperienceData
De les données au réseaux, de lui faire son apprentissage et de le sauvegarder
L'évaluation du modèle ce fait dans un autre fichier

"""

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout

generalDataPath = "../BatchExp/SecondBatchExp/DataExperience/"
trainDataPath = generalDataPath+"Train"
devDataPath = generalDataPath+"Dev"
neuralNetworkPath = generalDataPath + "NeuralNetwork"


typeTrain = "/train_300N1_300N2"
typeOfDev = "/dev_300N1_300N2"

#Load Data --------------------------------------------------------------------------------------

X_train = np.load(trainDataPath + typeTrain + "_X.npy")
y_train = np.load(trainDataPath + typeTrain + "_y.npy")
X_dev = np.load(devDataPath + typeOfDev + "_X.npy")
y_dev = np.load(devDataPath + typeOfDev + "_y.npy")

print("Data is loaded")
#Convert To tensor

X_train = tensorflow.convert_to_tensor(X_train)
y_train = tensorflow.convert_to_tensor(y_train)
X_dev = tensorflow.convert_to_tensor(X_dev)
y_dev = tensorflow.convert_to_tensor(y_dev)

#Construct the model------------------------------------------------------------------------------

#Construct Neural Network

model = Sequential()
model.add(Dropout(0.5,input_shape=(600,)))
model.add(Dense(480, activation='relu'))
model.add(Dense(300, activation='tanh'))

model.build()

model.summary()

model.compile(optimizer='adam', 
    loss="cosine_similarity")

my_callbacks = [tensorflow.keras.callbacks.EarlyStopping(patience=3)]
print("Fit data")

model.fit([[X_train]],[[y_train]],epochs=300,validation_data=(X_dev,y_dev),callbacks=my_callbacks)
model.save(neuralNetworkPath + "/NN_Hide=480_Train=_300N1_300N2_W2VF")


"""

https://www.tensorflow.org/tutorials/keras/keras_tuner

J'ai utilisé ce tutoriel pour optimiser mon nombre de couche'
"""

import kerastuner as kt

def model_builder(hp):
  model = Sequential()
  model.add(Dropout(0.1,input_shape=(600,)))
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(Dense(units=hp_units, activation='relu'))
  model.add(Dense(units=hp_units, activation='relu'))
  model.add(Dense(units=hp_units, activation='relu'))
  model.add(Dense(units=hp_units, activation='relu'))
  model.add(Dense(units=hp_units, activation='relu'))
  model.add(Dense(300))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer="adam",
                loss="cosine_similarity",
                metrics=['accuracy'])

  return model
"""
tuner = kt.Hyperband(model_builder,
                     objective='loss',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')




stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=50,validation_data=(X_dev,y_dev), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
"""










