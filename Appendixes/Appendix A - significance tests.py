# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:58:58 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle


#%% load data, merge and reduce to passes
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)

passes = events[events['eventName'] == 'Pass']

#%%
passes = events[events['eventName'] == 'Pass']
passes_drew = passes[passes['outcome'] == 'drew']
passes_status = passes[passes['status'] != 'drawing']

#%%
passes = events[events['eventName'] == 'Pass']
passes = passes[passes['outcome'] != 'drew']

#%%
passes_drawing = passes[passes['status'] == 'drawing']
#%%
ratio = 0.5

passes_unique = passes.drop_duplicates(subset = 'matchId')
passes_unique['matchBalance'] = np.abs((passes_unique['percHome'] - (passes_unique['percAway'] + 0.15)))
passes_unique_sorted_even = passes_unique.sort_values(axis = 0, by = 'matchBalance')
passes_unique_ratio = passes_unique_sorted_even.head(int(len(passes_unique_sorted_even)*ratio))
matches_even = pd.Series(passes_unique_ratio['matchId'].unique())
passes_even = passes.loc[passes['matchId'].isin(matches_even)]


#%%
passes_even_drawing = passes_even[passes_even['status'] == 'drawing']


#%%

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

#%%
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

#%%

perms_status = MC_perm_status(passes_status)

#%%
poss_status = perms_status[0] / (perms_status[0] + perms_status[1])
sns.histplot(poss_status)
plt.xlabel('Poss. of Trailing Teams')
plt.show()
                
#%%
p_status = get_empirical_p(poss_status, 0.521)

#%%
with open('perms_status.pickle', 'wb') as handle:
        pickle.dump(perms_status, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%

perms = MC_perm(passes)
poss = perms[0] / (perms[0] + perms[1])
print('Done')

perms_drawing = MC_perm(passes_drawing)
poss_drawing = perms_drawing[0] / (perms_drawing[0] + perms_drawing[1])
print('Done')

perms_even = MC_perm(passes_even)
poss_even = perms_even[0] / (perms_even[0] + perms_even[1])
print('Done')

perms_even_drawing = MC_perm(passes_even_drawing)
poss_even_drawing = perms_even_drawing[0] / (perms_even_drawing[0] + perms_even_drawing[1])
print('Done')


#%%
sns.histplot(poss)
plt.xlabel('Poss. of Winners')
plt.show()

sns.histplot(poss_drawing)
plt.xlabel('Poss. of Winners')
plt.show()

sns.histplot(poss_even)
plt.xlabel('Poss. of Losers')
plt.show()

sns.histplot(poss_even_drawing)
plt.xlabel('Poss. of Winners')
plt.show()

#%%get quantiles


def get_empirical_p(poss, obs_poss, N = 10000):
    poss_sorted = np.sort(poss)
    position_ind = np.searchsorted(poss_sorted, obs_poss)
    p = 1 - (position_ind/N)
    return p
    


#%%
p_outcome = get_empirical_p(poss, 0.528)
p_drawing = get_empirical_p(poss_drawing, 0.561)
p_even = get_empirical_p(poss_even, 0.521)
p_even_drawing = get_empirical_p(poss_even_drawing, 0.527)



#%%passes

with open('perms.pickle', 'wb') as handle:
        pickle.dump(perms, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('perms_drawing.pickle', 'wb') as handle:
    pickle.dump(perms_drawing, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save average scores per inner fold
with open('perms_even.pickle', 'wb') as handle:
    pickle.dump(perms_even, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('perms_even_drawing.pickle', 'wb') as handle:
    pickle.dump(perms_even_drawing, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
passes_head = passes.head(1000)