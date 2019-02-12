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

# ==========================================================================

class parameters : 
    def __init__(self):
        """Return the setting of paramters."""

        #-----------------------------------------------
        #----------- Network Parameters ----------------
        #-----------------------------------------------
        self.freq = 8
        self.offset = 0.8
        #            sin(freq*xx)+offset
        #            sin(freq*xx)-offset


        #----------- activation function ---------------
        # activation function can be selected here, the 
        # possible inputs are "sigmoid", "ReLU" and "Softplus"

        self.activation="sigmoid"

        #---------------- network size -----------------
        # the size of network can be specified here; note that 
        # it will be fully connected network, e.g. [2,2,2,2,2,2]
        # contains 6 layers with 2 nodes in every layer

        self.FC1 = 2
        self.FC2 = 2
        self.FC3 = 2
        self.FC4 = 2
        self.FC5 = 2
        self.FC6 = 2    
        self.sizeNet =[self.FC1,self.FC2,self.FC3,self.FC4,self.FC5,self.FC6]
        dimensionSet = [2*self.FC1, self.FC1, self.FC1*self.FC2, self.FC2,self.FC2*self.FC3, 
                        self.FC3,self.FC3*self.FC4, self.FC4, self.FC4*self.FC5, self.FC5, self.FC5*self.FC6, self.FC6]

        self.nv = np.sum(dimensionSet) # dimension of the problem
        #-----------------------------------------------
        #-----------------------------------------------

        #-----------------------------------------------
        #----------- Algorithm Parameters --------------
        #-----------------------------------------------

        self.seed = 17# random seed
        self.numIter = 1000 # maximum number of iteration
        self.mmr = 10 # memory for Sampled-SR1, Sampled-L_BFGS
        self.radius = 0.01 # radius for finding samples in Sampled-SR1, Sampled-L_BFGS

        ######## RUN alg with ########
        self.eps = 1e-3 # for updating matrix Bk

        self.eta = 1e-6 # used for ared/pred > eta

        self.deltak = 10 # initial TR radius

        self.epsTR = 1e-10 # used in condition in CG_Steinhaug

        self.cArmijo = 1e-4 # Armijo parameter

        self.rhoArmijo = .5 # backtracking factor

        self.GPUnumber = "0"

        self.startWithSampS_LBFGS = 0 # if you wanna sample from beginning in S-LBFGS algorithm put it 1, else 0 
        #-----------------------------------------------
        #-----------------------------------------------
