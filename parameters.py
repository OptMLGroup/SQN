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
        
        #----------- inputs for SIN problem ------------
        self.freq = 8
        self.offset = 0.8
        self.num_pts = 50
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

        self.num_weights = np.sum(dimensionSet) # dimension of the problem
        #-----------------------------------------------
        #-----------------------------------------------

        #-----------------------------------------------
        #----------- Algorithm Parameters --------------
        #-----------------------------------------------

        self.seed = 67            # random seed
        self.numIter = 1000       # maximum number of iterations
        self.mmr = 10             # memory length for S-LSR1, S-LBFGS
        self.radius = 1           # sampling radius for S-LSR1, S-LBFGS
        self.eps = 1e-8           # tolerance for updating quasi-Newton matrices
        self.eta = 1e-6           # tolerance for ared/pred reduction in TR
        self.delta_init = 1       # initial TR radius
        self.alpha_init = 2       # initial step length
        self.epsTR = 1e-10        # tolernace for CG_Steinhaug
        self.cArmijo = 1e-4       # Armijo sufficient decrease parameter
        self.rhoArmijo = .5       # Armijo backtracking factor
        self.init_sampling_SLBFGS = "off" # S-LBFGS sampling from first iteration
        #-----------------------------------------------
        #-----------------------------------------------
        
        #-----------------------------------------------
        #------------- Other Parameters ----------------
        #-----------------------------------------------
        self.GPUnumber = "0"      # GPU ID
