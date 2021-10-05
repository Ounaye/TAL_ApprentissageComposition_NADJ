#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:08:05 2021

@author: yannis.coutouly
"""

"""
This file is for testing to make a custom Model 
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

loss_tracker = keras.metrics.Mean(name="loss")
class testModel(keras.Model):

    
    def compile(self, optimizer, my_loss):
        super().compile(optimizer)
        self.my_loss = my_loss

    def train_step(self, data):
        layer_inNoun1 = data[0]["inNoun"]
        layer_inNoun2 = data[0]["inNoun2"]
        layer_inAdj1 = data[0]["inAdj"]
        layer_inAdj2 = data[0]["inAdj2"]
        with tf.GradientTape() as tape:
            y_pred = self(data[0], training=True)
            loss_value = self.my_loss(y_pred,layer_inNoun1,layer_inAdj1,layer_inNoun2,layer_inAdj2)
        
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
         
        loss_tracker.update_state(loss_value)
         
        
        return {"loss": loss_tracker.result()}
     
    def test_step(self, data):
        layer_inNoun1 = data[0]["inNoun"]
        layer_inNoun2 = data[0]["inNoun2"]
        layer_inAdj1 = data[0]["inAdj"]
        layer_inAdj2 = data[0]["inAdj2"]
        layer_inNAdj1 = self.get_layer("nadj").output #Not the good way
        layer_inNAdj2 = self.get_layer("nadj2").output #Not the good way
        y_pred = self(data[0], training=False)
        loss_value = self.my_loss(y_pred,layer_inNoun1,layer_inAdj1,layer_inNoun2,layer_inAdj2)
        loss_tracker.update_state(loss_value)
        
        return {"loss": loss_tracker.result()}

