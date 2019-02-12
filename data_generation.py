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
def getData(NN = 50, freq=8.0,offset=0.8):
    """Get and return the data."""
    
    xx = np.array(range(NN))*1.0/(NN+.0)
    xxa = np.sin(freq*xx)+offset
    xxb = np.sin(freq*xx)-offset

    X = [ [xx.tolist()+xx.tolist()],[xxa.tolist()+xxb.tolist()]]
    X = np.reshape(np.array(X),[2,-1]) 

    Y = [1 for _ in xrange(NN)]
    Y = Y+ [0 for _ in xrange(NN)]
    
    ns = len(Y)
    Y =  np.array(Y)
    X = np.transpose(X)

    
    
    X = np.array(X)
    y = np.array(Y)
    # y = np.reshape(y, newshape=(-1, 1))

    y = np.zeros([ns,2])
    for i in xrange(ns):
        y[i,Y[i]] = 1

    return X,Y,y   



