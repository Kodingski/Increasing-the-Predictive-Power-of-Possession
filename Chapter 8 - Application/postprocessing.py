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
import numpy as np
import copy

from utilities.utils import *
from utilities.model import *
from utilities.zones import *


#%%load final model
final_model = load_final_model()

#%%load matches
matches = load_matches()

#%% load passes and assign models zones
passes = load_passes()

#mirror passes and reduce to bottom half
passes = mirror_passes(passes)
passes = passes[passes['start_y'] <= 50]

#assign zones based on kmeans of final model
passes = assign_zones(passes, final_model['kmeans'])

#%%helper to prepare matches to input into model
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
    
            counts_status = passes_per_team_per_status.groupby(['zone'])['id'].count()
    
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

#%%Get models prediction for match
#put passes into format of model input
index_series = pd.Series(np.zeros(k), index = range(k))

X, match_info = prepare_matches(passes, mean_poss_vector, index_series, k)

#predict matches with final model
match_info[:,4] = final_model['model'].predict(X)

#%%combine dataframe with models final prediction and the mean centred possession per zone
matches_merged = np.hstack((match_info, X))
zone_names = [status + ' ' + str(zone) for status in ['trailing', 'drawing', 'leading'] for zone in range(k)]

#%%Create df enriched with match info
matches_overview = pd.DataFrame(matches_merged, columns = (['matchId', 'home_score', 'away_score', 'raw poss. Home', 'pred Home'] + zone_names))
matches_overview = pd.merge(matches[['wyId', 'label']], matches_overview, how = 'left', left_on = 'wyId', right_on = 'matchId')

#%%save to pickle
with open('matches_overview.pickle', 'wb') as handle:
    pickle.dump(matches_overview, handle, protocol=pickle.HIGHEST_PROTOCOL)
