# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:58:58 2021

@author: timsc
"""

#%% imports
import seaborn as sns
import matplotlib as plt

from utilities.utils import *
from utilities.plot_utils import *
from utilities.zones import *

#%% load passes
passes = load_passes()

#%%create different subsets
passes_drawing = passes[passes['status'] == 'drawing']
passes_even = passes.loc[passes['muBalance'] == 'even']
passes_even_drawing = passes_even[passes_even['status'] == 'drawing']

#%%helper to do permutation test
def MC_perm(passes, N = 10000, label = 'rand_outcome'):
    passes_red = passes[['matchId']]
    won_count_boot = np.zeros((N))
    lost_count_boot = np.zeros((N))
    
    matches = pd.DataFrame(passes['matchId'].unique())
    
    for i in range(N):
        if i%1000==0:
            print(i)
        

        matches[label] = np.random.randint(2, size=len(matches))
        passes_merged = passes_red.merge(matches, how = 'left', 
                                         left_on = 'matchId', right_on = 0)

        won_count_boot[i] = len(passes_merged[passes_merged[label] == 1])

        lost_count_boot[i] = len(passes_merged[passes_merged[label] == 0])
                
    return ([won_count_boot, lost_count_boot])

#%%helper to do permutation test for match status
def MC_perm_status(passes, N = 10000):
    passes_red = passes[['matchId', 'status']]
    trailing_count_boot = np.zeros((N))
    leading_count_boot = np.zeros((N))
        
    for i in range(N):
        if i%100==0:
            print(i)
        
        
        for _, passes_match in passes.groupby('matchId'):
            
        
            
            draw = np.random.randint(2, size = 1)
                
            if draw == 1:
                trailing_count_boot[i] += len(passes_match[passes_match['status'] == 'trailing'])
                leading_count_boot[i] += len(passes_match[passes_match['status'] == 'leading'])
                
            if draw == 0:
                trailing_count_boot[i] += len(passes_match[passes_match['status'] == 'leading'])
                leading_count_boot[i] += len(passes_match[passes_match['status'] == 'trailing'])
                
                
    return ([trailing_count_boot, leading_count_boot])


                
#%%do permutation tests for different subsets
perms = MC_perm(passes)

perms_status = MC_perm_status(passes)

perms_drawing = MC_perm(passes_drawing)

perms_even = MC_perm(passes_even)

perms_even_drawing = MC_perm(passes_even_drawing)
