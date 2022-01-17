# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 21:03:44 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from utilities.fitting import * 
from utilities.utils import *
from utilities.zones import *

import warnings
warnings.filterwarnings("ignore")


#%% load passes
passes = load_passes()

#%%mirror passes and reduce to bottom half
passes = mirror_passes(passes)
passes= passes[passes['start_y'] <= 50]

#%% define outer and inner cv
#number of splits inner outer
n_outer_splits = 5
n_inner_splits = 5

#initialite N-CV with seed 420
outer_cv = KFold(n_splits = n_outer_splits, shuffle = True, random_state = 420)
inner_cv = KFold(n_splits = n_inner_splits, shuffle = True, random_state = 420)

#%%  prepare dictionaries to store results in 
#store vor train results
models = {}
scores = {}
scores_averages = {}

#store rect train results
models_rect = {}
scores_rect = {}
scores_averages_rect = {}

#store vor test results
best_params = {}
best_models = {}
results_best_models = {}
kmeans_best_models = {}

#store rect test results
best_params_rect = {}
best_models_rect = {}
results_best_models_rect = {}

#store baseline test results
results_baseline_raw = {}
models_baseline_raw = {}
params_baseline_raw = {}
results_baseline_status = {}
models_baseline_status = {}
params_baseline_status = {}

#%%Run N-CV 
#As there are 25*5*5 models to be fit and evaluated, 
#only run this code on good hardware and with sufficient time.

#extract match ids to split on 
matches = pd.Series(passes['matchId'].unique())

#number of zones to check for and step size for k
ks = 20
k_step = 1

#outer loop
i = 0
for train_ix, test_ix in outer_cv.split(matches):
    
    models[i] = {}
    scores[i] = {}
    scores_average = {}
    
    models_rect[i] = {}
    scores_rect[i] = {}
    scores_average_rect = {}

    train_matches = matches.iloc[train_ix]
    test_matches = matches.iloc[test_ix]
    
    passes_train_outer = passes.loc[passes['matchId'].isin(train_matches)]
    passes_test = passes.loc[passes['matchId'].isin(test_matches)]
    
    #inner loop
    ii = 0
    for inner_train, inner_val in inner_cv.split(train_matches):
        
        
        print('*********************************')
        print(f'Outer fold: {i}, Inner fold {ii}')
        print('*********************************')
        
        inner_train_matches = train_matches.iloc[inner_train]
        val_matches = train_matches.iloc[inner_val]
        
        passes_train = passes.loc[passes['matchId'].isin(inner_train_matches)]
        passes_val = passes.loc[passes['matchId'].isin(val_matches)]
        
        
        models, scores, scores_average = train_fold(passes_train, passes_val, 
                                                    models, scores, scores_average, 
                                                    ) 
        
        passes_train = passes.loc[passes['matchId'].isin(inner_train_matches)]
        passes_val = passes.loc[passes['matchId'].isin(val_matches)]
        
        models_rect, scores_rect, scores_average_rect = train_fold(passes_train, passes_val, 
                                                    models_rect, scores_rect, 
                                                    scores_average_rect, zone_type = 'Rectangular'
                                                    ) 
        
        ii += 1

        
    scores_averages[i] = scores_average
    scores_averages_rect[i] = scores_average_rect

    #find best hyperparams of the fold        
    best_model_voron = ['Voronoi'] + find_best_model_outer_fold(scores_average)
    best_model_rect = ['Rectangular'] + find_best_model_outer_fold(scores_average_rect)
    
    #test best full model fold
    best_params[i] = best_model_voron
    print(f'Best Model Vor: {best_model_voron}')
    best_models[i], results_best_models[i], kmeans_best_models[i] = fit_and_score_best_model(passes_train_outer, 
                                                                                            passes_test,
                                                                                            best_params[i])
    print(f'Test score: {np.round(results_best_models[i],4)}')


    best_params_rect[i] = best_model_rect   
    print(f'Best Model Rect: {best_model_rect}')
    passes_train_outer = passes.loc[passes['matchId'].isin(train_matches)]
    passes_test = passes.loc[passes['matchId'].isin(test_matches)]    
    
    best_models_rect[i], results_best_models_rect[i] = fit_and_score_best_model(passes_train_outer, 
                                                             passes_test,
                                                             best_params_rect[i])
    print(f'Test score: {np.round(results_best_models_rect[i],4)}')


    
    #train best models found fkr baselines on outer train set and test on outer test
    #train baselines raw, k = 1
    passes_train_outer = passes.loc[passes['matchId'].isin(train_matches)]
    passes_test = passes.loc[passes['matchId'].isin(test_matches)]
    params_baseline_raw[i] = ['Baseline'] + find_best_model_outer_fold(scores_average,
                                                              baseline = True,
                                                              status_type='raw')
    models_baseline_raw[i], results_baseline_raw[i] = fit_and_score_best_model(passes_train_outer, 
                                                            passes_test,
                                                            params_baseline_raw[i])
        
    
    # train baselines status, k = 1
    passes_train_outer = passes.loc[passes['matchId'].isin(train_matches)]
    passes_test = passes.loc[passes['matchId'].isin(test_matches)]
    params_baseline_status[i] = ['Baseline'] + find_best_model_outer_fold(scores_average,
                                                              baseline = True,
                                                              status_type='status')
    models_baseline_status[i], results_baseline_status[i] = fit_and_score_best_model(passes_train_outer, 
                                                            passes_test,
                                                            params_baseline_status[i])
    

    i += 1
    
    # save results
    save_results()       

