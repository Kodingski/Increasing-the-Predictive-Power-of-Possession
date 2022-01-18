# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:01:59 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd

from utilities.utils import *

import warnings
warnings.filterwarnings("ignore")
#%% load all results dicts
model, results = load_full_results()

#%% Transform data into matrix form
#get mean results best models and put results into matrix
results_matrix = np.zeros((1,6))
for i, params in enumerate(results['params_baseline_status'].values()):
   
   results_matrix[0,i] = np.round(params[4], 3)
   
results_matrix[0,5] = np.round(np.mean(results_matrix[0,:5]),3)

#create data matrix folds/k
matrix = np.zeros((25,20))
scores = results['scores']

for i, outer_fold in enumerate(scores.values()):
    for ii, inner_fold in enumerate(outer_fold.values()):
        for k, score in inner_fold.items():
            
            row = i*5 + ii 
            k = int(k-1)
            matrix[row, k] = np.round(score['status']['full'],2)


#%% Table C.1 - Full Train Results Vor.
index_folds = []
for i in range(1,6,1):
    index_folds = index_folds + list(np.round(np.arange(i+ .1, i + 0.6,0.1), 1))


matrix_df = pd.DataFrame(matrix, index = index_folds, columns = range(1,21))


latex = matrix_df.transpose().to_latex()

#%%Repeat the same for rectangular zones
matrix_rect = np.zeros((25,6))
scores_rect = results['scores_rect']

for i, outer_fold in enumerate(scores_rect.values()):
    for ii, inner_fold in enumerate(outer_fold.values()):
        iii = 0
        for k, score in inner_fold.items():
            
            row = i*5 + ii 
            k = int(k-1)
            matrix_rect[row, iii] = np.round(score['status']['full'],2)
            iii += 1
            
            
#%%Table C.2 - Full Train Results Vor.
index_folds = []
for i in range(1,6,1):
    index_folds = index_folds + list(np.round(np.arange(i+ .1, i + 0.6,0.1), 1))


matrix_df_rect = pd.DataFrame(matrix_rect, index = index_folds, columns = [1,2,6,8,15,18])
latex_rect = matrix_df_rect.transpose().to_latex()



