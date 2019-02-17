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

import numpy as np
import pickle
import os.path
import os
import sys
import tensorflow as tf
import time
from util_func import *
from network import *
from data_generation import *
from sampleSY import *

# ==========================================================================
def S_LBFGS(w_init,X,y,seed,numIter,mmr,radius,eps,alpha_init,cArmijo,rhoArmijo,num_weights,init_sampling_SLBFGS,dnn,sess):
    """Sampled LBFGS method."""  
    
    w = w_init
    sess.run(dnn.params.assign(w))                        # Assign initial weights to parameters of the network
    np.random.seed(seed)                                  # Set random seed
    
    print(seed)
    numFunEval = 0                                        # Initialize counters (function values, gradients and Hessians)
    numGradEval = 0
    numHessEval = 0
    
    gamma_k = 1

    g_kTemp, objFunOldTemp = sess.run( [dnn.G,[dnn.cross_entropy,dnn.accuracy]] , feed_dict={dnn.x: X, dnn.y:y})
    numFunEval += 1
    numGradEval += 1
    objFunOld = objFunOldTemp[0]
    acc = objFunOldTemp[1]  
    g_k = g_kTemp[0]
    norm_g =  LA.norm( g_k ) 

    HISTORY = []
    weights_SLBFGS = []

    k=0
    st=time.time()
    
    alpha = alpha_init

    while 1:
         
        weights_SLBFGS.append(sess.run(dnn.params))
        
        HISTORY.append([k, objFunOld,acc,norm_g, numFunEval,numGradEval,numHessEval, numFunEval+numGradEval+numHessEval,
                        time.time()-st,alpha])
        
        print HISTORY[k]                                   # Print History array
        
        if k > numIter or acc ==1:                         # Terminate if number of iterations > numIter or Accuracy = 1
            break
        
        if init_sampling_SLBFGS == "off" and k == 0:
            alpha = min(1,1.0/(np.linalg.norm(g_k, ord=1)))
            pk = g_k
        else:
            S,Y,counterSucc,numHessEval,gamma_k = sample_pairs_SY_SLBFGS(X,y,num_weights,mmr,radius,eps,dnn,numHessEval,sess)
            pk = L_BFGS_two_loop_recursion(g_k,S,Y,k,mmr,gamma_k,num_weights)
            alpha = 2*alpha   # change to 2*alpha

        mArmijo = -(pk.T.dot(g_k))
        
        x0 = sess.run(dnn.params)
        while 1:
          # params is the updated variable by adding -alpha* pk  to the previous one  
          sess.run(dnn.updateOp, feed_dict={dnn.updateVal:  -alpha* pk  })   

          objFunNew = sess.run(dnn.cross_entropy, feed_dict={dnn.x: X, dnn.y:y})
          numFunEval += 1
          if objFunOld + alpha*cArmijo* mArmijo < objFunNew :
            sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: x0})
            alpha = alpha * rhoArmijo
            if alpha < 1e-25:
              print "issue with Armijo"
              break
          else:
            break
        objFunOld = objFunNew

        xNew, acc, g_k_newTemp = sess.run( [dnn.params,dnn.accuracy, dnn.G] , feed_dict={dnn.x: X, dnn.y:y}) 
        numGradEval += 1
        g_k = g_k_newTemp[0]
        norm_g =  LA.norm( g_k ) 
        k += 1

        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xNew})
        
    pickle.dump( HISTORY, open( "./_saved_log_files/S_LBFGS.pkl", "wb" ) )                    # Save History in .pkl file
    # pickle.dump( weights_SLBFGS, open( "./_saved_log_files/S_LBFGS_weights.pkl", "wb" ) )    # Save Weights in .pkl file
