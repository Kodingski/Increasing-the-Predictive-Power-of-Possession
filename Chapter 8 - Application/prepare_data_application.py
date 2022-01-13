# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:54:30 2021

@author: timsc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:19:26 2021

@author: timsc
"""
#%% imports
import os
import copy
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
import matplotlib.patches as patches
import random
import statsmodels.api as sm
import pickle


from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *

#%% load data, merge and reduce to passes
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)       

#%%helpers
def mirror_passes(passes):
    passes_mirrored = passes.copy()
    passes_mirrored['start_y'] = 100 - passes_mirrored['start_y']
    passes_merged = passes.append(passes_mirrored)
    return(passes_merged)
    

def assign_zones(passes, kmeans):

    passes['zone'] = kmeans.predict(passes[['start_x', 'start_y']])       
    
    return(passes)

def get_mean_poss_vector(passes, k, status = True):
    
    passes_home = passes[passes['home'] == passes['teamId']]
    passes_away = passes[passes['home'] != passes['teamId']]
    
    
    if status:
        pass_count_per_zone_per_status_home = passes_home.groupby(['status','zone'])['eventId'].count().values
        #rearrange to have trailing first
        pass_count_per_zone_per_status_home = np.hstack(
                                        (pass_count_per_zone_per_status_home[(2*k):(3*k)],
                                        pass_count_per_zone_per_status_home[0:(2*k)])
                                        )
        
        
        pass_count_per_zone_per_status_away = passes_away.groupby(['status','zone'])['eventId'].count()
        #rearrange to have trailing first
        pass_count_per_zone_per_status_away = np.hstack(
                                        (pass_count_per_zone_per_status_away[(k):(2*k)],
                                        pass_count_per_zone_per_status_away[0:(k)],
                                        pass_count_per_zone_per_status_away[(2*k):(3*k)])
                                        )
    
        mean_poss_vector = pass_count_per_zone_per_status_home / (pass_count_per_zone_per_status_home +
                                                             pass_count_per_zone_per_status_away)
        
    else:
        pass_count_per_zone_home = passes_home.groupby(['zone'])['eventId'].count().values
        pass_count_per_zone_away = passes_away.groupby(['zone'])['eventId'].count().values
        
        mean_poss_vector = pass_count_per_zone_home / (pass_count_per_zone_home +
                                                             pass_count_per_zone_away)

    return mean_poss_vector


#%%load results
folder_results = 'results/final/'

files = os.listdir(folder_results)
results = {}


for f in files:
    if f == 'models.pickle':
        models = pickle.load(open(folder_results+f, "rb"))
    else:                        
        results[str(f.rstrip('.pickle'))] = pickle.load(open(folder_results+f, "rb"))
        

final_zones, final_model = results['final_kmeans'], results['final_mod']      


#%%
passes = events[events['eventName'] == 'Pass']
passes = mirror_passes(passes)
passes = passes[passes['start_y'] <= 50]

passes = assign_zones(passes, final_zones)


#%%create dataframe containing match id and X per match

#%%

def prepare_matches(passes, mean_poss_vector, index_series, k):
    n = len(passes['matchId'].unique())
    
    X = np.zeros((n, (k*3)))
    match_info = np.zeros((n, 5))
    

    passes_per_match = passes.groupby(['matchId'])  


    i = 0    
    for match, passes_match in passes_per_match:
      
        mean_poss_vector_copy = copy.deepcopy(mean_poss_vector).reshape((1,(3*k)))
    
        counts_home = np.zeros((k*3)) 
        counts_away = np.zeros((k*3)) 
        
        home_team = passes_match['home'].unique()[0]
    
        per_team_per_status = passes_match.groupby(['teamId', 'status'])
            
        
        for name, passes_per_team_per_status in per_team_per_status:
            team, status = name[0], name[1]
    
            counts_status = passes_per_team_per_status.groupby(['zone'])['eventId'].count()
    
            if team == home_team:
                
                if status == 'trailing':
                    counts_home[0:k] = counts_status.add(index_series, fill_value = 0).values
                
                if status == 'drawing':
                    counts_home[k:(2*k)] = counts_status.add(index_series, fill_value = 0).values
                    
                if status == 'leading':
                    counts_home[(2*k):(3*k)] = counts_status.add(index_series, fill_value = 0).values
    
            else:
                  
                #reverse order here as if home is trailing, away is leading and vice versa
                if status == 'trailing':
                    counts_away[(2*k):(3*k)] = counts_status.add(index_series, fill_value = 0).values
                
                if status == 'drawing':
                    counts_away[k:(2*k)] = counts_status.add(index_series, fill_value = 0).values
                        
                if status == 'leading':
                    counts_away[0:k] = counts_status.add(index_series, fill_value = 0).values
       
        weight_vector = counts_home + counts_away
        
        #divide, and for case of division by 0 (no passes played in zone, default to mean vector)
    
        
        poss_vector_draw = np.divide(counts_home[k:(2*k)], (weight_vector[k:(2*k)]),
                                            out=mean_poss_vector_copy[0,k:(2*k)],
                                            where=weight_vector[k:(2*k)] != 0)
        
        poss_vector_draw = poss_vector_draw - mean_poss_vector[k:(2*k)]
        poss_vector_default = np.hstack((poss_vector_draw, poss_vector_draw, poss_vector_draw)).reshape((1,(3*k)))
        
        
        poss_vector = np.divide(counts_home, (weight_vector),
                                            out=np.zeros((1,(k*3))),
                                            where=weight_vector != 0)
    
                
        poss_vector_mean_con = np.subtract(poss_vector, mean_poss_vector, 
                                            out = poss_vector_default, 
                                            where = weight_vector != 0)
    
    
    
        X[i,:] = poss_vector_mean_con
           
        
        
        match_info[i,0] = int(match)
        match_info[i,1] = passes_match.loc[passes_match['teamId'] == home_team, 'own_score'].max() 
        match_info[i,2] = passes_match.loc[passes_match['teamId'] == home_team, 'opp_score'].max()
        match_info[i,3] = np.sum(counts_home) / (np.sum(counts_home) + np.sum(counts_away))

        
        i += 1
        
    return(X, match_info)


#%%
k = 11
mean_poss_vector = get_mean_poss_vector(passes, k)

#%%
passes = assign_zones(passes, final_zones)

#%%
index_series = pd.Series(np.zeros(k), index = range(k))

X, match_info = prepare_matches(passes, mean_poss_vector, index_series, k)

#%%
match_info[:,4] = final_model.predict(X)

#%%build dataframe
matches_merged = np.hstack((match_info, X))
zone_names = [status + ' ' + str(zone) for status in ['trailing', 'drawing', 'leading'] for zone in range(k)]


matches_overview = pd.DataFrame(matches_merged, columns = (['matchId', 'home_score', 'away_score', 'raw poss. Home', 'pred Home'] + zone_names))

#%%load matches 

matches = pd.read_json(f"data\matches\matches_all_leagues_odds.json", encoding='unicode_escape')

#%%
matches_overview = pd.merge(matches[['wyId', 'label']], matches_overview, how = 'left', left_on = 'wyId', right_on = 'matchId')


#%%save to pickle
with open('matches_overview.pickle', 'wb') as handle:
    pickle.dump(matches_overview, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('mean_poss_vector.pickle', 'wb') as handle:
    pickle.dump(matches_overview, handle, protocol=pickle.HIGHEST_PROTOCOL)