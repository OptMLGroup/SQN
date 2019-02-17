#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019 Albert Berahas, Majid Jahani, Martin Takáč
# 
# All Rights Reserved.
# 
# Authors: Albert Berahas, Majid Jahani, Martin Takáč
# 
# Please cite:
# 
#   A. S. Berahas, M. Jahani, and M. Takáč, "Quasi-Newton Methods for 
#   Deep Learning: Forget the Past, Just Sample." (2019). Lehigh University.
#   http://arxiv.org/abs/1901.09997
# ==========================================================================

import tensorflow as tf
import numpy as np
import time

# ==========================================================================
def weight_variable(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std, dtype=tf.float64)
    return tf.Variable(initial,dtype=tf.float64)


# ==========================================================================
class DNN:
    """This class constructs the network used.
    The inputs are: (1) sizeNet (total number of weights, 
    and (2) activation. 
    Note, that for this code, we fix the network to have
    6 fully connected layers, with varying number of nodes.
    Moreover, we concatenate the weight matrices into a long
    vector as this allows for easier implementation of our 
    methods."""
    def __init__(self,hiddenSizes,activation="sigmoid"):

        x = tf.placeholder(tf.float64, shape=[None, 2])
        y_ = tf.placeholder(tf.float64, shape=[None, 2])

        FC1 = hiddenSizes[0]
        FC2 = hiddenSizes[1]
        FC3 = hiddenSizes[2]
        FC4 = hiddenSizes[3]
        FC5 = hiddenSizes[4]
        FC6 = 2

        sizes = [2*FC1, FC1, FC1*FC2, FC2,FC2*FC3, FC3,FC3*FC4, FC4, FC4*FC5, FC5, FC5*FC6, FC6]

        n = np.sum(sizes)
        params = weight_variable([n, 1],1.0/(n))
        uparam = tf.unstack(params,axis = 0)

        W1 = tf.reshape(tf.stack(  uparam[0:sizes[0]] ), shape=[2,FC1])
        b1 = tf.reshape(tf.stack(  uparam[sum(sizes[0:1]):sum(sizes[0:1])+sizes[1]] ), shape=[FC1])

        W2 = tf.reshape(tf.stack(  uparam[sum(sizes[0:2]):sum(sizes[0:2])+sizes[2]] ), shape=[FC1, FC2])
        b2 = tf.reshape(tf.stack(  uparam[sum(sizes[0:3]):sum(sizes[0:3])+sizes[3]] ), shape=[FC2])

        W3 = tf.reshape(tf.stack(  uparam[sum(sizes[0:4]):sum(sizes[0:4])+sizes[4]] ), shape=[FC2, FC3])
        b3 = tf.reshape(tf.stack(  uparam[sum(sizes[0:5]):sum(sizes[0:5])+sizes[5]] ), shape=[FC3])

        W4 = tf.reshape(tf.stack(  uparam[sum(sizes[0:6]):sum(sizes[0:6])+sizes[6]] ), shape=[FC3, FC4])
        b4 = tf.reshape(tf.stack(  uparam[sum(sizes[0:7]):sum(sizes[0:7])+sizes[7]] ), shape=[FC4])

        W5 = tf.reshape(tf.stack(  uparam[sum(sizes[0:8]):sum(sizes[0:8])+sizes[8]] ), shape=[FC4, FC5])
        b5 = tf.reshape(tf.stack(  uparam[sum(sizes[0:9]):sum(sizes[0:9])+sizes[9]] ), shape=[FC5])

        W6 = tf.reshape(tf.stack(  uparam[sum(sizes[0:10]):sum(sizes[0:10])+sizes[10]] ), shape=[FC5, FC6])
        b6 = tf.reshape(tf.stack(  uparam[sum(sizes[0:11]):sum(sizes[0:11])+sizes[11]] ), shape=[FC6])

        Ws = [W1,W2,W3,W4,W5,W6]
        bs = [b1,b2,b3,b4,b5,b6]
        
        
        if activation=="sigmoid":
            acf = tf.nn.sigmoid
        if activation=="ReLU":
            acf = tf.nn.relu
        if activation=="Softplus":
            acf = tf.nn.softplus

        a1 = acf(tf.matmul(x, W1) + b1)
        a2 = acf(tf.matmul(a1, W2) + b2)
        a3 = acf(tf.matmul(a2, W3) + b3)
        a4 = acf(tf.matmul(a3, W4) + b4)
        a5 = acf(tf.matmul(a4, W5) + b5)
        a6 = (tf.matmul(a5, W6) + b6)

        
        #-----------------------------------------------
        #----------- Function, Gradient, Hessian, Accuracy and Other Operators --------------
        #-----------------------------------------------
        output = a6                               # Output of network
        probdist = tf.nn.softmax(output)              # Softmax of output layer 
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))        # Cross entropy loss
        correct_prediction = tf.equal(tf.argmax(a6, 1), tf.argmax(y_,1))                 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))             # Accuracy computation

        self.output= output
        self.probdist=probdist
        self.x = x
        self.y = y_
        self.Ws = Ws
        self.bs = bs
        self.cross_entropy = cross_entropy
        self.accuracy = accuracy
        self.correct_prediction = correct_prediction
        self.params = params
        
        self.updateVal = tf.placeholder(tf.float64, shape=[int(params.shape[0]),1])      # Placeholder for updating parameters 
        self.updateOp = tf.assign_add(params, self.updateVal).op                         # Operator for updating parameters  
        self.G = tf.gradients(cross_entropy,params)                                      # Gradient computation
        self.H = tf.hessians(cross_entropy,params)                                       # Hessian computation
        self.ASSIGN_OP = tf.assign(self.params, self.updateVal).op                       # Operator for assigning parameters
