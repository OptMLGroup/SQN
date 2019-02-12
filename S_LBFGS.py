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

# ==========================================================================
def S_LBFGS(freq,offset,activation,sizeNet,w,seedd,numIter,mmr,radius,eps,cArmijo,rhoArmijo,GPUnumber,nv,startWithSamp):
    """Sampled LBFGS method."""  
    X,Y,y = getData(50,freq, offset)
    os.environ["CUDA_VISIBLE_DEVICES"] =  GPUnumber 
    sess = tf.InteractiveSession()
    dnn = DNN(sizeNet,activation)
    w = np.reshape(w,(nv,1))
    sess.run(dnn.params.assign(w))
    np.random.seed(seedd)
    print "Norm of initial point is: ", LA.norm(w)
    numFunEval = 0
    numGradEval = 0
    numHessEval = 0
    # Initial value for updating initial Hessian Approx matrix
    gamma_k = 1

    g_kTemp, objFunOldTemp= sess.run( [dnn.G,[dnn.cross_entropy,dnn.accuracy]] , feed_dict={dnn.x: X, dnn.y:y})
    numFunEval += 1
    objFunOld = objFunOldTemp[0]
    acc = objFunOldTemp[1]

    numGradEval += 1
    g_k = g_kTemp[0]
    norm_g =  LA.norm( g_k ) 

    counter= 0
    HISTORY = []
    varLBFGSsampled = []
    minsTy = 0
    maxsTy = 0
    k=0
    alpha = 2
    alphaa = alpha
    st=time.time()
    while 1:
        varLBFGSsampled.append(sess.run(dnn.params))
        
        if startWithSamp == 0:
            # just gradient step for k = 0
            if k == 0:
              alpha = min(1,1.0/(np.linalg.norm(g_k, ord=1)))
              pk = g_k
            else:
              pk = L_BFGS_two_loop_recursion(g_k,S,Y,k,mmr,gamma_k,nv)

        if startWithSamp == 1:
            Stmp = np.random.randn(nv,mmr)
            Hess = np.squeeze(sess.run( [dnn.H] , feed_dict={dnn.x: X, dnn.y:y}))
            Ytmp = np.matmul(Hess,Stmp)
            S = np.zeros((nv,0))
            numHessEval += 1
            Y = np.zeros((nv,0))
            sample = 0
            minsTy = 1e+20
            maxsTy = 0    
            for jj in xrange(mmr):
              sTy = Ytmp[:,jj].T.dot(Stmp[:,jj])
              if sTy > eps *(LA.norm(Stmp[:,jj])*LA.norm(Ytmp[:,jj])):
                  if sTy < minsTy:
                      minsTy = sTy
                  if sTy > maxsTy:
                      maxsTy = sTy
                  gamma_k = np.squeeze((Stmp[:,jj]).T.dot(Ytmp[:,jj])/((Ytmp[:,jj]).T.dot(Ytmp[:,jj])))
                  S = np.append(S,Stmp[:,jj].reshape(nv,1),axis = 1)
                  Y = np.append(Y,Ytmp[:,jj].reshape(nv,1),axis=1)
                  sample += 1
            pk = L_BFGS_two_loop_recursion(g_k,S,Y,k,mmr,gamma_k,nv)


        mArmijo = -(pk.T.dot(g_k))

        HISTORY.append([k, objFunOld,acc,norm_g, numFunEval,numGradEval,numHessEval, numFunEval+numGradEval+numHessEval,
                        time.time()-st,minsTy,maxsTy,alphaa])
        if np.mod(k,1) == 0 or acc ==1:
          print HISTORY[k]

        if k > numIter or acc ==1 :
          break    
        x0 = sess.run(dnn.params)
        while 1:
          # params is the updated variable by adding -alpha* pk  to the previous one  
          sess.run(dnn.updateOp, feed_dict={dnn.updateVal:  -alpha* pk  })   

          objFunNew = sess.run(dnn.cross_entropy, feed_dict={dnn.x: X, dnn.y:y})
          numFunEval += 1
          if objFunOld < objFunNew-alpha*cArmijo* mArmijo :
            sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: x0})
            alpha = alpha * rhoArmijo
            if alpha < 1e-25:
              print "issue with Armijo"
              break
          else:
            break
        objFunOld = objFunNew
        alphaa = alpha    
        # xNew is the updated var satisfying in Armijo condition  
        xNew, acc, g_k_newTemp = sess.run( [dnn.params,dnn.accuracy, dnn.G] , feed_dict={dnn.x: X, dnn.y:y}) 
        numGradEval += 1
        g_k_new = g_k_newTemp[0]
        norm_g =  LA.norm( g_k_new ) 
        k += 1
        numSamples = mmr

        if startWithSamp == 0:
            Stmp = np.random.randn(nv,mmr)
            Hess = np.squeeze(sess.run( [dnn.H] , feed_dict={dnn.x: X, dnn.y:y}))
            Ytmp = np.matmul(Hess,Stmp)
            S = np.zeros((nv,0))
            numHessEval += 1
            Y = np.zeros((nv,0))
            sample = 0
            minsTy = 1e+20
            maxsTy = 0    
            for jj in xrange(mmr):
              sTy = Ytmp[:,jj].T.dot(Stmp[:,jj])
              if sTy > eps *(LA.norm(Stmp[:,jj])*LA.norm(Ytmp[:,jj])):
                  if sTy < minsTy:
                      minsTy = sTy
                  if sTy > maxsTy:
                      maxsTy = sTy
                  gamma_k = np.squeeze((Stmp[:,jj]).T.dot(Ytmp[:,jj])/((Ytmp[:,jj]).T.dot(Ytmp[:,jj])))
                  S = np.append(S,Stmp[:,jj].reshape(nv,1),axis = 1)
                  Y = np.append(Y,Ytmp[:,jj].reshape(nv,1),axis=1)
                  sample += 1


        g_k = g_k_new
#         alpha = alpha*2
        alpha = 2
        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xNew})
    pickle.dump( HISTORY, open( "./_saved_log_files/S_LBFGS.pkl", "wb" ) )
