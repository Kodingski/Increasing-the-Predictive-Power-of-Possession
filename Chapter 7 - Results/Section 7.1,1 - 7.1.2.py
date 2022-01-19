# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:01:59 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utilities.utils import *
from utilities.fitting import *
import warnings
warnings.filterwarnings("ignore")
#%% load all results dicts
model, results = load_full_results()

#%% Preprocess results data
#create data matrix folds/k
matrix = np.zeros((25,20))
scores = results['scores']

for i, outer_fold in enumerate(scores.values()):
    for ii, inner_fold in enumerate(outer_fold.values()):
        for k, score in inner_fold.items():
            
            row = i*5 + ii 
            k = int(k-1)
            matrix[row, k] = np.round(score['status']['full'],2)

#transform into dataframe
index_folds = []
for i in range(1,6,1):
    index_folds = index_folds + list(np.round(np.arange(i+ .1, i + 0.6,0.1), 1))

matrix_df = pd.DataFrame(matrix, index = index_folds, columns = range(1,21))

#do the same for rect
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
            
index_folds = []
for i in range(1,6,1):
    index_folds = index_folds + list(np.round(np.arange(i+ .1, i + 0.6,0.1), 1))

matrix_df_rect = pd.DataFrame(matrix_rect, index = index_folds, columns = [1,2,6,8,15,18])
#%%create dataframe for sns.lineplot
stacked = matrix_df.stack()
long = stacked.reset_index()
long.columns = ['Fold', 'k', 'Val. Score']
long['Zone Type'] = 'Voronoi'

rect_stacked = matrix_df_rect.stack()
rect_long = rect_stacked.reset_index()
rect_long.columns = ['Fold', 'k', 'Val. Score']
rect_long['Zone Type'] = 'Rectangular'

#merge dfs and split folds in seperate columsn
full_long = long.append(rect_long)  
full_long[['Outer Fold', 'Inner Fold']] = full_long['Fold'].astype(str).str.split('.', 1, expand = True)

#%%Figure 7.1 - Test Scores per Inner Loop for both zone types visualzied
for name, group in full_long.groupby('Outer Fold'):
    
    
    ax_fold = sns.lineplot(x = 'k', y = 'Val. Score', data = group, hue = 'Zone Type', ci=None,
             markers = True, style = 'Zone Type')
    ax_fold.set_xticks(range(20+1))
    ax_fold.set_yticks(np.arange(0.15, 0.23, 0.01))
    ax_fold.set_ylabel('Mean Val. Score')

    plt.show()
    
#%%Table 7.1 and 7.2 - Test Scores inner Loops
#create table test scores rect
print(results['best_params_rect'])

#create table test scores vor
print(results['best_params'])

#%%get mean results best models and put results into matrix
results_matrix = np.zeros((1,6))

#loop over results 
for i, params in enumerate(results['params_baseline_status'].values()):
   results_matrix[0,i] = np.round(params[4], 3)
   
results_matrix[0,5] = np.round(np.mean(results_matrix[0,:5]),3)

latex_rect = matrix_df_rect.transpose().to_latex()

#%%Table 7.3 - Test Scores Outer Loop vs baseline
# calculate mean r_squared test outer
calc_mean_r_squared_outer(results['results_baseline_raw'])
calc_mean_r_squared_outer(results['results_baseline_status'])
calc_mean_r_squared_outer(results['results_best_models']) 

print(np.round(list(results['results_baseline_raw'].values()),3))
print(np.round(list(results['results_baseline_status'].values()),3))
print(np.round(list(results['results_best_models'].values()),3))




