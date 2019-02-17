#!/usr/bin/enum_weights python
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
def S_LSR1(w_init,X,y,seed,numIter,mmr,radius,eps,eta,delta_init,epsTR,num_weights,dnn,sess):
    """Sampled LSR1 method."""       
    
    w = w_init
    sess.run(dnn.params.assign(w))                        # Assign initial weights to parameters of the network
    np.random.seed(seed)                                  # Set random seed
    
    numFunEval = 0                                        # Initialize counters (function values, gradients and Hessians)
    numGradEval = 0
    numHessEval = 0
    
    deltak = delta_init                                   # Initialize trust region radius

    HISTORY = []                                          # Initialize array for storage
    weights_SLSR1 = []                                    # Initialize array for storing weights           

    k=0                                                   # Initialize iteration counter   
    st = time.time()                                      # Start the timer

    objFunOld = sess.run(dnn.cross_entropy,feed_dict={dnn.x: X, dnn.y:y})    # Compute function value at current iterate
    numFunEval += 1
    
    print objFunOld
    
    # Method while loop (terminate after numIter or Accuracy 1 achieved)
    while 1:
        gradTemp, acc, xOld = sess.run([dnn.G,dnn.accuracy,dnn.params], 
                         feed_dict={dnn.x: X, dnn.y:y})                      # Compute gradient and accuracy
        gard_k = gradTemp[0]
        numGradEval += 1
        norm_g = LA.norm(gard_k)

        # Sample S, Y pairs
        S,Y,counterSucc,numHessEval = sample_pairs_SY_SLSR1(X,y,num_weights,mmr,radius,eps,dnn,numHessEval,sess)

        # Append to History array
        HISTORY.append([k, objFunOld,acc,norm_g,numFunEval,numGradEval,numHessEval,numFunEval+numGradEval+numHessEval,
                        counterSucc,time.time()-st,deltak])
        print HISTORY[k]                                   # Print History array
        
        if k > numIter or acc ==1:                         # Terminate if number of iterations > numIter or Accuracy = 1
            break
        
        weights_SLSR1.append(sess.run(dnn.params))        # Append weights


        sk_TR = CG_Steinhaug_matFree(epsTR, gard_k , deltak,S,Y,num_weights)         # Compute step using CG Steinhaug
        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xOld + sk_TR })            # Assign new weights

        objFunNew = sess.run(dnn.cross_entropy, feed_dict={dnn.x: X, dnn.y:y})       # Compute new function value
        numFunEval += 1
        
        ared = objFunOld - objFunNew                     # Compute actual reduction             

        Lp = np.zeros((Y.shape[1],Y.shape[1]))
        for ii in xrange(Y.shape[1]):
           for jj in range(0,ii): 
                  Lp[ii,jj] = S[:,ii].dot(Y[:,jj])
        tmpp = np.sum((S * Y),axis=0)
        Dp = np.diag(tmpp)
        Mp = (Dp + Lp + Lp.T)
        Minvp = np.linalg.inv(Mp) 
        tmpp1 = np.matmul(Y.T,sk_TR)
        tmpp2 = np.matmul(Minvp,tmpp1)
        Bk_skTR = np.matmul(Y,tmpp2)    
        pred = -(gard_k.T.dot(sk_TR) + 0.5* sk_TR.T.dot(Bk_skTR))          # Compute predicted reduction

        # Take step 
        if ared/pred > eta:
          xNew = xOld + sk_TR
          objFunOld = objFunNew
        else:
          xNew = xOld

        # Update trust region radius
        if ared/pred > 0.75:
            deltak = 2*deltak
        elif ared/pred>=0.1 and ared/pred <=0.75:
          pass # no need to change deltak
        elif ared/pred<0.1:
          deltak = deltak*0.5

        k += 1                                                         # Increment iteration counter
        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xNew})       # Assign updated weights
        
    pickle.dump( HISTORY, open( "./_saved_log_files/S_LSR1.pkl", "wb" ) )                    # Save History in .pkl file
    # pickle.dump( weights_SLSR1, open( "./_saved_log_files/S_LSR1_weights.pkl", "wb" ) )    # Save Weights in .pkl file
    