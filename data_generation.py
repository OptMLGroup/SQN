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
def getData(num_pts = 50, freq = 8.0, offset = 0.8):
    """Get and return the data."""
    
    # Create array with num_pts points between 0 and 1 (i.e., 0,1/num_pts, 2/num_pts,...)
    xx = np.array(range(num_pts))*1.0/(num_pts+.0)
    # Create positive (xp) and negative (xn) classes
    xp = np.sin(freq*xx)+offset
    xn = np.sin(freq*xx)-offset

    # Concatenate the two arrays into list and reshape
    X = [ [xx.tolist()+xx.tolist()],[xp.tolist()+xn.tolist()]]
    X = np.reshape(np.array(X),[2,-1]) 

    # Create labels Y
    Y = [1 for _ in xrange(num_pts)]
    Y = Y + [0 for _ in xrange(num_pts)]

    ns = len(Y)
    Y = np.array(Y)
    X = np.transpose(X)
    y = np.zeros([ns,2])
    for i in xrange(ns):
        y[i,Y[i]] = 1

    return X,y   



