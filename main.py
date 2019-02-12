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
import os.path

# ==========================================================================
def main(opt):
    """Call the selected solver with the selected parameters."""    
    if opt == "SLSR1":
        S_LSR1(cp.freq,cp.offset,cp.activation,cp.sizeNet,w,cp.seed,cp.numIter,cp.mmr,
               cp.radius,cp.eps,cp.eta,cp.deltak,cp.epsTR,cp.GPUnumber,cp.nv)
    elif opt == "SLBFGS":
        S_LBFGS(cp.freq,cp.offset,cp.activation,cp.sizeNet,w,cp.seed,cp.numIter,cp.mmr,
                cp.radius,cp.eps,cp.cArmijo,cp.rhoArmijo,cp.GPUnumber,cp.nv,cp.startWithSampS_LBFGS)
       
# calling the parameters
cp = parameters()

cp.seed = 67

# Initial point
np.random.seed(cp.seed)
w= np.random.randn(cp.nv,1)

# ==========================================================================

if __name__ == '__main__':
    """Run the selected solver."""  
    main("SLBFGS")