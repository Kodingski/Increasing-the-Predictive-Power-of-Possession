# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:33:01 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd
import random
import copy
import pickle

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


import warnings
warnings.filterwarnings("ignore")




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
    

def get_mean_poss_vector(passes_train, k, status = True):
    
    passes_home = passes_train[passes_train['home'] == passes_train['teamId']]
    passes_away = passes_train[passes_train['home'] != passes_train['teamId']]
    
    
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

def prepare_data(passes_train, mean_poss_vector, mean_poss_vector_raw, index_series, k):
    
    n = len(passes_train['matchId'].unique())
    
    X_raw = np.zeros((n,k))
    X = np.zeros((n, (k*3))) 
    y = []

    passes_per_match = passes_train.groupby(['matchId'])   
     
    i = 0    
    for match, passes_match in passes_per_match:
        
        mean_poss_vector_copy = copy.deepcopy(mean_poss_vector).reshape((1,(3*k)))
        mean_poss_vector_raw_copy = copy.deepcopy(mean_poss_vector_raw).reshape((1,k))

        counts_raw_home = np.zeros((k))
        counts_raw_away = np.zeros((k))
        counts_home = np.zeros((k*3)) 
        counts_away = np.zeros((k*3)) 
        
        home_team = passes_match['home'].unique()[0]

        per_team_per_status = passes_match.groupby(['teamId', 'status'])
        
        
        for name, passes_per_team_per_status in per_team_per_status:
            team, status = name[0], name[1]

            counts_status = passes_per_team_per_status.groupby(['zone'])['eventId'].count()

            if team == home_team:
                
                counts_raw_home = counts_raw_home + counts_status.add(index_series, fill_value = 0).values
                
                if status == 'trailing':
                    counts_home[0:k] = counts_status.add(index_series, fill_value = 0).values
                
                if status == 'drawing':
                    counts_home[k:(2*k)] = counts_status.add(index_series, fill_value = 0).values
                    
                if status == 'leading':
                    counts_home[(2*k):(3*k)] = counts_status.add(index_series, fill_value = 0).values

            else:
                
                counts_raw_away = counts_raw_away + counts_status.add(index_series, fill_value = 0).values
  
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

        poss_raw = np.divide(counts_raw_home, (counts_raw_home + counts_raw_away),
                                        out=mean_poss_vector_raw_copy,
                                        where=(counts_raw_home + counts_raw_away) != 0)



        X[i,:] = poss_vector_mean_con
        X_raw[i,:] = poss_raw - mean_poss_vector_raw
           
        score_diff = (passes_match.loc[passes_match['teamId'] == home_team, 'own_score'].max() 
                      - passes_match.loc[passes_match['teamId'] == home_team, 'opp_score'].max())
        
        y.append(score_diff)
        
        i += 1
        
    return(X_raw, X, y)


def assign_vor_zones(passes_train, passes_test, k):
    
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_train[['start_x', 'start_y']])
    passes_train['zone'] = kmeans.predict(passes_train[['start_x', 'start_y']])       
    passes_test['zone'] = kmeans.predict(passes_test[['start_x', 'start_y']])
    
    return(passes_train, passes_test, kmeans)

def fit_model(passes, k, raw = False, intercept_only = False):
    
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes[['start_x', 'start_y']])
    passes['zone'] = kmeans.predict(passes[['start_x', 'start_y']])       
    
    
    mean_poss_vector = get_mean_poss_vector(passes, k)
    mean_poss_vector_raw = get_mean_poss_vector(passes, k, status=False)
    index_series = pd.Series(np.zeros(k), index = range(k))
    
    X_raw, X, y = prepare_data(passes, mean_poss_vector, mean_poss_vector_raw, index_series, k)



    if intercept_only:
        X = np.ones((len(y), 1))
        model_sk = LinearRegression().fit(X, y)
        model = OLS(y, add_constant(X)).fit()
        return(X, y, model, model_sk)

    if raw: 
        
        model_sk = LinearRegression().fit(X_raw, y)
        model = OLS(y, add_constant(X_raw)).fit()
        return (X_raw, y, model, model_sk)
    else:
        model_sk = LinearRegression().fit(X,y)
        model = OLS(y, add_constant(X)).fit()
        return(X, y, model, model_sk, kmeans)
    
    
    
def get_AIC(X, y, model):
    y_hat = model.predict(X)
    resid = y - y_hat
    sse = sum(resid**2)
    k= len(model.coef_)
    AIC= 2*k - 2*np.log(sse)
    
    return (AIC)

def get_BIC(X, y, model):
    y_hat = model.predict(X)
    resid = y - y_hat
    sse = sum(resid**2)
    k = len(model.coef_)
    n = len(X)
    BIC = (n*np.log(sse/n)) + (k*np.log(n))

    return (BIC)
#%%
passes = events[events['eventName'] == 'Pass']
passes = mirror_passes(passes)
passes = passes[passes['start_y'] <= 50]

#%%intercept only model
X, y, int_model, int_model_sk = fit_model(passes, 1, intercept_only = True)
print(int_model.summary())

#%%fit baselines
X_raw, y, final_baseline_model_raw, final_baseline_model_raw_sk = fit_model(passes, 1, raw=True)
print(final_baseline_model_raw.aic)
print(final_baseline_model_raw.summary())
#%%
X_base, y_base, final_baseline_model_status, final_baseline_model_status_sk, k_means_base = fit_model(passes, 1)
print(final_baseline_model_status.aic)
print(final_baseline_model_status.summary())
#%%
#fit full model
X, y, final_model, final_model_s, kmeans = fit_model(passes, 11)
print(final_model.aic)
print(final_model.summary())

#%%
for table in final_model.summary().tables:
    print(table)

#%%
predictions = final_model_sk.predict(X)

#%%
sorted_prediction = np.where(predictions == np.amax(predictions))

#%%
games = passes.groupby('matchId')['matchId'].unique()
#%%
match_highest_pred = games.iloc[388]


passes_2500694 = passes[passes['matchId'] == 2500694]

predictions_perc = 1/(1 + np.exp(-(predictions/2)))

#%%save final model

