# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:47:51 2022

@author: timsc
"""

#%%
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *


#%%define countries to include
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']


#%% load data, merge and reduce to passes

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events_country['country'] = country
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)

#%% reduce to passes for quicker computation
passes = events[events['eventName'] == 'Pass']


#%%Find *ratio* amount of most even matches
ratio = 0.5

matches_unique = passes.drop_duplicates(subset = 'matchId')
matches_unique['matchBalance'] = np.abs((matches_unique['percHome'] - (matches_unique['percAway'] + 0.15)))
matches_unique_sorted_even = matches_unique.sort_values(axis = 0, by = 'matchBalance')
matches_unique_ratio = matches_unique_sorted_even.head(int(len(matches_unique_sorted_even)*ratio))

#%% create class variable 'muBalance' to differentiate between even and uneven half of matches
events.loc[events['matchId'].isin(matches_unique_ratio['matchId']),'muBalance'] = 'even'
events.loc[~events['matchId'].isin(matches_unique_ratio['matchId']),'muBalance'] = 'uneven'


#%% reduce to variables used and discussed in the thesis project
events_thesis = events[['id', 'matchId', 'teamId', 'playerId', 'home',
                       'muBalance', 'outcome', 'eventName', 'subEventName',
                       'matchPeriod', 'eventSec', 'own_score', 'opp_score', 'status', 'start_x', 'start_y',
                       'country'
                       ]]


#%%
events_thesis_test = events_thesis.sample(1000)

#%%split back into leagues, for smaller files

for country in countries:
    

    events_country = events_thesis.loc[events_thesis['country'] == country]
    events_country = events_country.drop('country', axis = 1)
    events_country.to_json(f'data/preprocessed/events_{country}_thesis.json')



