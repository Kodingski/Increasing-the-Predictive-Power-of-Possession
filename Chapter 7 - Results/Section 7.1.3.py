# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:59:25 2021

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
passes= passes_mirrored[passes_mirrored['start_y'] <= 50]

#%% define outer and inner cv
#number of splits inner outer
n_outer_splits = 5
n_inner_splits = 5

#initialite N-CV with seed 420
outer_cv = KFold(n_splits = n_outer_splits, shuffle = True, random_state = 420)
inner_cv = KFold(n_splits = n_inner_splits, shuffle = True, random_state = 420)

#%%
matches_even = pd.Series(passes_unique_ratio['matchId'].unique())

#%%
n_outer_splits = 5
n_inner_splits = 5

outer_cv = KFold(n_splits = n_outer_splits, shuffle = True, random_state = 420)
inner_cv = KFold(n_splits = n_inner_splits, shuffle = True, random_state = 420)



#%%  test on even subset#
#extract match ids to split on 
matches = passes['matchId'].unique()

#extract match ids of even matches to test on 
matches_even = passes.loc[passes['muBalance' == 'even']]['matchId'].unique()

#create np array to store test results in
test_mu_balance = np.zeros((6,5))
i = 0
for train_ix, test_ix in outer_cv.split(matches):

    row = 0
    train_matches = matches.iloc[train_ix]
    
    test_matches = matches.iloc[test_ix]
    test_matches = test_matches[test_matches.isin(matches_even)]
    
    passes_train_outer = passes.loc[passes['matchId'].isin(train_matches)]
    passes_test_even = passes.loc[passes['matchId'].isin(test_matches)]
    passes_test_uneven = passes.loc[~passes['matchId'].isin(test_matches)]



    _, test_mu_balance[row,i] = fit_and_score_best_model(passes_train_outer, 
                                                                passes_test_even,
                                                                results['params_baseline_raw'][0])
    
    
    print(f'Test score Baseline Raw even: {np.round(test_mu_balance[row,i],4)}')                         
    row+=1
    _, test_mu_balance[row,i] = fit_and_score_best_model(passes_train_outer, 
                                                                passes_test_uneven,
                                                                results['params_baseline_raw'][0])
    print(f'Test score Baseline Raw uneven: {np.round(test_mu_balance[row,i],4)}') 
    row+=1





    _, test_mu_balance[row,i] = fit_and_score_best_model(passes_train_outer, 
                                                                passes_test_even,
                                                                results['params_baseline_status'][0])
    print(f'Test score Baseline Status even: {np.round(test_mu_balance[row,i],4)}')                         
    row+=1

    _, test_mu_balance[row,i] = fit_and_score_best_model(passes_train_outer, 
                                                                passes_test_uneven,
                                                                results['params_baseline_status'][0])
    print(f'Test score Baseline Status uneven: {np.round(test_mu_balance[row,i],4)}') 
    row+=1



    _, test_mu_balance[row,i], _ = fit_and_score_best_model(passes_train_outer, 
                                                                                            passes_test_even,
                                                                                            results['best_params'][0])
    print(f'Test score Voronoi k = 11, even : {np.round(test_mu_balance[row,i],4)}')                         
    row+=1


    _, test_mu_balance[row,i], _ = fit_and_score_best_model(passes_train_outer, 
                                                            passes_test_uneven,
                                                            results['best_params'][0])
    print(f'Test score Voronoi k = 11, uneven: {np.round(test_mu_balance[row,i],4)}')                         
    row+=1                                                                                        


    i += 1

#%%
with open('test_mu_balance.pickle', 'wb') as handle:
    pickle.dump(test_mu_balance, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%Turn into Dataframe and latex



Columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']




MU_even_results = pd.DataFrame(test_mu_balance[(0,2,4),], columns = Columns)
MU_even_results['Mean'] = MU_even_results.mean(axis=1)
MU_even_results = np.round(MU_even_results, 3)

MU_even_results_latex = MU_even_results.to_latex()


MU_uneven_results = pd.DataFrame(test_mu_balance[(1,3,5),], columns = Columns)
MU_uneven_results['Mean'] = MU_uneven_results.mean(axis=1)
MU_uneven_results = np.round(MU_uneven_results, 3)
MU_uneven_results_latex = MU_uneven_results.to_latex()
