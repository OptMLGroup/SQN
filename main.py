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
import matplotlib.pyplot as plt
import pickle
from S_LSR1 import *
from S_LBFGS import *
from parameters import *
from network import *
from data_generation import *
import os.path
import sys

input1 = sys.argv[1]


# ==========================================================================
def main(opt=input1):
    
    """Call the selected solver with the selected parameters."""    
    if opt == "SLSR1":
        S_LSR1(w_init,X,y,cp.seed,cp.numIter,cp.mmr,cp.radius,cp.eps,cp.eta,cp.delta_init,cp.epsTR,cp.num_weights,dnn,sess)
    elif opt == "SLBFGS":
        S_LBFGS(w_init,X,y,cp.seed,cp.numIter,cp.mmr,
                cp.radius,cp.eps,cp.alpha_init,cp.cArmijo,cp.rhoArmijo,cp.num_weights,cp.init_sampling_SLBFGS,dnn,sess)
       
# Get the parameters
cp = parameters()

# Create the data
X,y = getData(cp.num_pts,cp.freq,cp.offset)

# Create network
os.environ["CUDA_VISIBLE_DEVICES"] = cp.GPUnumber 
sess = tf.InteractiveSession()
dnn = DNN(cp.sizeNet,cp.activation)

# Set the initial point
np.random.seed(cp.seed)
w_init = np.random.randn(cp.num_weights,1)

# ==========================================================================
if __name__ == '__main__':
    """Run the selected solver."""  
    main()