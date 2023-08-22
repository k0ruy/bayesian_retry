#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:47:09 2022

@author: giorgio
"""
import numpy as np
import pandas as pd

# decides whether to move to the proposed island
def next_position(current,proposal):
    next_vis = current
    if proposal>current:
        next_vis = proposal
    else:
        if np.random.uniform()< (proposal/current):
            next_vis = proposal
    return (next_vis)

# sampling the islands
pop = range(1,8)
start = 3
samples = 50000
position = np.ones([samples,1])
position[0] = start

for i in range(1,samples):
    # proposal = right, remaining within the allowed bounds
    proposal = min(position[i-1]+1,7)

    #with probability 50%,  proposal is changed to left (negative values are forbidden)
    if  np.random.uniform()<0.5: 
        proposal = max(position[i-1]-1,1)
    
    #this array is the trace of the sampled values    
    position[i] = next_position(position[i-1],proposal)
    
    
df = pd.DataFrame(position, columns=["position"])
#contingency table
freqs = pd.crosstab(index=df['position'],columns=df['position'], margins=True)["All"][0:7]
empirical_p = freqs / sum(freqs)
true_p = range(1,8)/np.sum(range(1,8))
