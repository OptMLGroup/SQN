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
import random
from numpy import linalg as LA
import math

# ==========================================================================
def CG_Steinhaug_matFree(epsTR, g , deltak, S,Y,nv):
  """
  The following function is used for sloving the trust region subproblem
  by utilizing "CG_Steinhaug" algorithm discussed in 
  Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.; 
  moreover, for Hessian-free implementation, we used the compact form of Hessian
  approximation discussed in Byrd, Richard H., Jorge Nocedal, and Robert B. Schnabel. 
  "Representations of quasi-Newton matrices and their use in limited memory methods." 
  Mathematical Programming 63.1-3 (1994): 129-156
  """
  zOld = np.zeros((nv,1))
  rOld = g
  dOld = -g
  trsLoop = 1e-12
  if LA.norm(rOld) < epsTR:
    return zOld
  flag = True
  pk= np.zeros((nv,1))

# for Hessfree
  L = np.zeros((Y.shape[1],Y.shape[1]))
  for ii in xrange(Y.shape[1]):
     for jj in range(0,ii): 
            L[ii,jj] = S[:,ii].dot(Y[:,jj])
 
    
  tmp = np.sum((S * Y),axis=0)

  D = np.diag(tmp)
  M = (D + L + L.T)
  Minv = np.linalg.inv(M)

  while flag:
    
    ################
    tmp1 = np.matmul(Y.T,dOld)
    tmp2 = np.matmul(Minv,tmp1)
    Bk_d = np.matmul(Y,tmp2)
    
    ################

    if dOld.T.dot(Bk_d) < trsLoop:
      tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
      pk = zOld + tau*dOld
      flag = False
      break
    alphaj = rOld.T.dot(rOld) / (dOld.T.dot(Bk_d))
    zNew = zOld +alphaj*dOld
    
    if LA.norm(zNew) >= deltak:
      tau = rootFinder(LA.norm(dOld)**2, 2*zOld.T.dot(dOld), (LA.norm(zOld)**2 - deltak**2))
      pk = zOld + tau*dOld
      flag = False
      break
    rNew = rOld + alphaj*Bk_d
    
    if LA.norm(rNew) < epsTR:
      pk = zNew
      flag = False
      break
    betajplus1 = rNew.T.dot(rNew) /(rOld.T.dot(rOld))
    dNew = -rNew + betajplus1*dOld
    
    zOld = zNew
    dOld = dNew
    rOld = rNew
  return pk


# ==========================================================================
def rootFinder(a,b,c):      
  """return the root of (a * x^2) + b*x + c =0"""
  r = b**2 - 4*a*c

  if r > 0:
      num_roots = 2
      x1 = ((-b) + np.sqrt(r))/(2*a+0.0)     
      x2 = ((-b) - np.sqrt(r))/(2*a+0.0)
      x = max(x1,x2)
      if x>=0:
        return x
      else:
        print "no positive root!"
  elif r == 0:
      num_roots = 1
      x = (-b) / (2*a+0.0)
      if x>=0:
        return x
      else:
        print "no positive root!"
  else:
      print("No roots")

def L_BFGS_two_loop_recursion(g_k,S,Y,k,mmr,gamma_k,nv):
  """
  The following function returns the serach direction based
  on LBFGS two loop recursion discussed in 
  Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.
  """    
#   idx = min(k,mmr)
  idx = min(S.shape[1],mmr)  
  rho = np.zeros((idx,1))
  
  theta = np.zeros((idx,1))
  q = g_k
  for i in xrange(idx):
    rho[idx-i-1] = 1/ S[:,idx-i-1].reshape(nv,1).T.dot(Y[:,idx-i-1].reshape(nv,1))
    theta[idx-i-1] =(rho[idx-i-1])*(S[:,idx-i-1].reshape(nv,1).T.dot(q))
    q = q - theta[idx-i-1]*Y[:,idx-i-1].reshape(nv,1)
    
  r = gamma_k*q
  for j in xrange(idx):
    beta = (rho[j])*(Y[:,j].reshape(nv,1).T.dot(r))
    r = r + S[:,j].reshape(nv,1)*(theta[j] - beta)
    
  return r
