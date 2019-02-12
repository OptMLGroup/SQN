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
def S_LSR1(freq,offset,activation,sizeNet,w,seedd,numIter,mmr,radius,eps,eta,deltakk,epsTR,GPUnumber,nv):
    """Sampled LSR1 method."""    
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
    deltak = 1

    HISTORY = []
    varSR1sampled = []

    k=0
    counter = 0
    st = time.time()

    S = np.zeros((nv,mmr))
    Y = np.zeros((nv,mmr))
    counterSucc = 0
    objFunOld = sess.run(dnn.cross_entropy,feed_dict={dnn.x: X, dnn.y:y})
    numFunEval += 1
    while 1:
        gradTemp, acc, xOld = sess.run([dnn.G,dnn.accuracy,dnn.params], 
                                                 feed_dict={dnn.x: X, dnn.y:y})
        gard_k = gradTemp[0]
        numGradEval += 1
        norm_g = LA.norm(gard_k)

        Hess_xNew = np.squeeze(sess.run(dnn.H, feed_dict={dnn.x: X, dnn.y:y}))

        Stemp = np.random.randn(nv,mmr)
        Ytemp = np.matmul(Hess_xNew,Stemp) # nv*mmr matrix
        numHessEval += 1
        S = np.zeros((nv,0))
        Y = np.zeros((nv,0))

        for idx in xrange(mmr):        

          L = np.zeros((Y.shape[1],Y.shape[1]))
          for ii in xrange(Y.shape[1]):
             for jj in range(0,ii): 
                    L[ii,jj] = S[:,ii].dot(Y[:,jj])


          tmp = np.sum((S * Y),axis=0)
          D = np.diag(tmp)
          M = (D + L + L.T)
          Minv = np.linalg.inv(M)        

          tmp1 = np.matmul(Y.T,Stemp[:,idx])
          tmp2 = np.matmul(Minv,tmp1)
          Bksk = np.squeeze(np.matmul(Y,tmp2))
          yk_BkskDotsk = (  Ytemp[:,idx]- Bksk ).T.dot(  Stemp[:,idx]  )  
          if np.abs(np.squeeze(yk_BkskDotsk)) > (
                 eps *(LA.norm(Ytemp[:,idx]- Bksk )  * LA.norm(Stemp[:,idx]))  ):        
            counterSucc += 1

            S = np.append(S,Stemp[:,idx].reshape(nv,1),axis = 1)
            Y = np.append(Y,Ytemp[:,idx].reshape(nv,1),axis=1)


        HISTORY.append([k, objFunOld,acc,norm_g,numFunEval,numGradEval,numHessEval,numFunEval+numGradEval+numHessEval,
                        counterSucc,time.time()-st,deltak])
        if k > numIter or acc ==1 :
          print HISTORY[k]
          break
        if np.mod(k,1) == 0:
          print HISTORY[k]
        varSR1sampled.append(sess.run(dnn.params))


        sk_TR = CG_Steinhaug_matFree(epsTR, gard_k , deltak,S,Y,nv) 
        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xOld + sk_TR })

        objFunNew = sess.run(dnn.cross_entropy, feed_dict={dnn.x: X, dnn.y:y})
        numFunEval += 1
        ared = objFunOld - objFunNew

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

        pred = -(gard_k.T.dot(sk_TR) + 0.5* sk_TR.T.dot(Bk_skTR))

        if ared/pred > eta:
          xNew = xOld + sk_TR
          objFunOld = objFunNew
        else:
          xNew = xOld

        if ared/pred > 0.75:
            deltak = 2*deltak

        elif ared/pred>=0.1 and ared/pred <=0.75:
          pass # no need to change deltak
        elif ared/pred<0.1:
          deltak = deltak*0.5

        counterSucc = 0

        k += 1 
        sess.run(dnn.ASSIGN_OP, feed_dict={dnn.updateVal: xNew})
    pickle.dump( HISTORY, open( "./_saved_log_files/S_LSR1.pkl", "wb" ) )
  