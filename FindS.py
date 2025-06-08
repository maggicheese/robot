# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:44:34 2025

@author: Dell
"""

import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('Training_examples.csv')
concepts = np.array(data)[:, :-1]  # Extract features

target = np.array(data)[:, -1]  # Extract target labels

def train(con, tar):
    specific_h = None  # Initialize specific hypothesis
    
    # Find the first positive example to initialize specific_h
    for i, val in enumerate(tar):
        if val == 'Yes':
            specific_h = con[i].copy()
            break
    
    if specific_h is None:
        return "No positive example found."
    
    # Generalize specific_h by iterating over the remaining examples
    for i, val in enumerate(con):
        if tar[i] == 'Yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
    
    return specific_h

# Print the final specific hypothesis
print(train(concepts, target))