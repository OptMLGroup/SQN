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
from numpy import linalg as LA
import tensorflow as tf

def sample_pairs_SY_SLSR1(X,y,num_weights,mmr,radius,eps,dnn,numHessEval,sess):
    """ Function that computes SY pairs for S-LSR1 method"""
    Hess = np.squeeze(sess.run(dnn.H, feed_dict={dnn.x: X, dnn.y:y}))
    numHessEval += 1
    Stemp = radius*np.random.randn(num_weights,mmr)
    Ytemp = np.matmul(Hess,Stemp) # nv*mmr matrix

    S = np.zeros((num_weights,0))
    Y = np.zeros((num_weights,0))

    counterSucc = 0
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

        S = np.append(S,Stemp[:,idx].reshape(num_weights,1),axis = 1)
        Y = np.append(Y,Ytemp[:,idx].reshape(num_weights,1),axis=1)
      
    return S,Y,counterSucc,numHessEval

def sample_pairs_SY_SLBFGS(X,y,num_weights,mmr,radius,eps,dnn,numHessEval,sess):
    """ Function that computes SY pairs for S-LBFGS method"""
    Hess = np.squeeze(sess.run( [dnn.H] , feed_dict={dnn.x: X, dnn.y:y}))
    numHessEval += 1
    
    Stemp = radius*np.random.randn(num_weights,mmr)
    Ytemp = np.matmul(Hess,Stemp)
    
    S = np.zeros((num_weights,0))
    Y = np.zeros((num_weights,0))
    
    counterSucc = 0  
    for idx in xrange(mmr):
      sTy = Ytemp[:,idx].T.dot(Stemp[:,idx])
      if sTy > eps *(LA.norm(Stemp[:,idx])*LA.norm(Ytemp[:,idx])):
          gamma_k = np.squeeze((Stemp[:,idx]).T.dot(Ytemp[:,idx])/((Ytemp[:,idx]).T.dot(Ytemp[:,idx])))
          S = np.append(S,Stemp[:,idx].reshape(num_weights,1),axis = 1)
          Y = np.append(Y,Ytemp[:,idx].reshape(num_weights,1),axis=1)
          counterSucc += 1      
    return S,Y,counterSucc,numHessEval,gamma_k
        