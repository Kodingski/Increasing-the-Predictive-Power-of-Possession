# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:47:43 2022

@author: timsc
"""

import numpy as np
import pandas as pd
import copy
import pickle

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, ElasticNet
import warnings
warnings.filterwarnings("ignore")

def mirror_passes(passes):
    passes_mirrored = passes.copy()
    passes_mirrored['start_y'] = 100 - passes_mirrored['start_y']
    passes_merged = passes.append(passes_mirrored)
    return(passes_merged)


def get_mean_poss_vector(passes_train, k, status = True):
    
    passes_home = passes_train[passes_train['home'] == passes_train['teamId']]
    passes_away = passes_train[passes_train['home'] != passes_train['teamId']]
    
    
    if status:
        pass_count_per_zone_per_status_home = passes_home.groupby(['status','zone'])['id'].count().values
        #rearrange to have trailing first
        pass_count_per_zone_per_status_home = np.hstack(
                                        (pass_count_per_zone_per_status_home[(2*k):(3*k)],
                                        pass_count_per_zone_per_status_home[0:(2*k)])
                                        )
        
        
        pass_count_per_zone_per_status_away = passes_away.groupby(['status','zone'])['id'].count()
        #rearrange to have trailing first
        pass_count_per_zone_per_status_away = np.hstack(
                                        (pass_count_per_zone_per_status_away[(k):(2*k)],
                                        pass_count_per_zone_per_status_away[0:(k)],
                                        pass_count_per_zone_per_status_away[(2*k):(3*k)])
                                        )
    
        mean_poss_vector = pass_count_per_zone_per_status_home / (pass_count_per_zone_per_status_home +
                                                             pass_count_per_zone_per_status_away)
        
    else:
        pass_count_per_zone_home = passes_home.groupby(['zone'])['id'].count().values
        pass_count_per_zone_away = passes_away.groupby(['zone'])['id'].count().values
        
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

            counts_status = passes_per_team_per_status.groupby(['zone'])['id'].count()

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


def assign_rect_zones(passes_train, passes_test, k):
    if k == 1:
        k_root = 1
    elif k == 6:
        k_root = 3
    elif k == 15:
        k_root = 5
    else:
        k_root = np.sqrt(k*2)
        
    step = 100/k_root
    
    zone = 0
    
    for x in np.arange(0, 100, step):
        for y in np.arange(0, 50, step):
            
            x = int(x)
            y = int(y)
            
            passes_train.loc[((passes_train.loc[:, 'start_x'].between(x, x+step)) & (passes_train.loc[:, 'start_y'].between(y, y+step))) ,'zone'] = zone      
            passes_test.loc[((passes_test.loc[:, 'start_x'].between(x, x+step)) & (passes_test.loc[:, 'start_y'].between(y, y+step))) ,'zone'] = zone                                       
            zone += 1  
            
            
    return(passes_train, passes_test)


def assign_vor_zones(passes_train, passes_test, k):
    
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_train[['start_x', 'start_y']])
    passes_train['zone'] = kmeans.predict(passes_train[['start_x', 'start_y']])       
    passes_test['zone'] = kmeans.predict(passes_test[['start_x', 'start_y']])
    
    return(passes_train, passes_test, kmeans)


def train_fold(passes_train, passes_val, models, scores, scores_average, 
               i, ii, ks, k_step, n_outer_splits, n_inner_splits, zone_type = 'Voronoi'):
        

    models[i][ii] = {}
    scores[i][ii] = {}

    for k in range(1, ks+1, k_step):
        
        print('*********************************')
        print(f'Zone type: {zone_type}, k: {k}')
        print('*********************************')
        
        #assign zones for rectangular zones
        if zone_type == 'Rectangular':
            
            #check if k is valid for rectangular zones, else continue
            if k not in np.array([1,2,6,8,15,18]):
                continue
            
            passes_train, passes_val = assign_rect_zones(passes_train, passes_val, k)
            
    
        models[i][ii][k] = {}
        models[i][ii][k]['status'] = {}
        models[i][ii][k]['raw'] = {}
        
        scores[i][ii][k] = {}
        scores[i][ii][k]['status'] = {}
        scores[i][ii][k]['raw'] = {}

        #assign zones for Voronoi zones
        if zone_type == 'Voronoi':
            passes_train, passes_val, models[i][ii][k]['kmeans'] = assign_vor_zones(passes_train, passes_val, k)


        if k not in scores_average.keys():
            scores_average[k] = {}
        
        if 'status' not in scores_average[k].keys():
            scores_average[k]['status'] = {}
            scores_average[k]['raw'] = {}

        mean_poss_vector = get_mean_poss_vector(passes_train, k)
        mean_poss_vector_raw = get_mean_poss_vector(passes_train, k, status = False)
        #create series of 0s with index of k zones to later add 
        #to passes per zone to have a way of dealing with zones without a pass played
        index_series = pd.Series(np.zeros(k), index = range(k))
        
        X_raw, X, y = prepare_data(passes_train, mean_poss_vector,
                                      mean_poss_vector_raw, index_series, k)
        
        
        #Cs = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
        Cs = []
    
        model_status = LinearRegression()
        model_status.fit(X, y)
        models[i][ii][k]['status']['full'] = model_status


        model_raw = LinearRegression()
        model_raw.fit(X_raw, y)
        models[i][ii][k]['raw']['full'] = model_raw

        for C in Cs: 

            model_regularized_status = ElasticNet(alpha = C/50)
            model_regularized_status.fit(X, y)
            models[i][ii][k]['status'][C] = model_regularized_status
       
            
            model_regularized_raw = ElasticNet(alpha = C/50)
            model_regularized_raw.fit(X_raw, y)                
            models[i][ii][k]['raw'][C] = model_regularized_raw


        #prepare Validation data
        X_raw_val, X_val, y_val = prepare_data(passes_val, mean_poss_vector,
                                      mean_poss_vector_raw, index_series, k)


        #save scores
        model_status_score = model_status.score(X_val, y_val)
        scores[i][ii][k]['status']['full'] = model_status_score            

        
        if 'full' not in scores_average[k]['status'].keys():
            scores_average[k]['status']['full'] = 0
        
        scores_average[k]['status']['full'] = scores_average[k]['status']['full']  + ((1/n_inner_splits) *
                                                                      model_status_score)            
        for C in Cs: 
            
            model_regularized_status_score = models[i][ii][k]['status'][C].score(X_val, y_val)
            scores[i][ii][k]['status'][C] = model_regularized_status_score            
            
            if C not in scores_average[k]['status'].keys():
                scores_average[k]['status'][C] = 0

            scores_average[k]['status'][C] = scores_average[k]['status'][C] + ((1/n_inner_splits) 
                                                           * model_regularized_status_score)
            
        #do it for raw
        model_raw_score = model_raw.score(X_raw_val, y_val)
        scores[i][ii][k]['raw']['full'] = model_raw_score            

             
        if 'full' not in scores_average[k]['raw'].keys():
            scores_average[k]['raw']['full'] = 0
        
        scores_average[k]['raw']['full'] = scores_average[k]['raw']['full']  + ((1/n_inner_splits) *
                                                                      model_raw_score)   

        for C in Cs: 
            
            model_regularized_raw_score = models[i][ii][k]['raw'][C].score(X_raw_val, y_val)
            scores[i][ii][k]['raw'][C] = model_regularized_raw_score            
            
            if C not in scores_average[k]['raw'].keys():
                scores_average[k]['raw'][C] = 0

            scores_average[k]['raw'][C] = scores_average[k]['raw'][C] + ((1/n_inner_splits) 
                                                           * model_regularized_raw_score)
    return (models, scores, scores_average)


def find_best_model_outer_fold(scores_average, baseline = False, 
                               status_type = False):

    
    best_found = -1000
    for k, model_type in scores_average.items():
        if baseline and k != 1:
            continue
        
        for model_type, alpha in model_type.items():
            if status_type == 'raw' and model_type != 'raw':
                continue
            if status_type == 'status' and model_type != 'status':
                continue
            
            
            for alpha, test_result in alpha.items():
                if test_result > best_found:
                    best_k = k
                    best_type = model_type
                    best_alpha = alpha
                    best_found = test_result
    return([best_k, best_type, best_alpha, best_found])    
    
    
def fit_and_score_best_model(passes_train_outer, passes_test, best_model):
    
    k = best_model[1]
    model_type = best_model[2]
    regularization = best_model[3]
    
    
    if best_model[0] == 'Voronoi':
        passes_train_outer, passes_test, k_means = assign_vor_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    if best_model[0] == 'Rectangular':
        
        passes_train_outer, passes_test = assign_rect_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    if best_model[0] == 'Baseline':
        passes_train_outer, passes_test = assign_rect_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    
    mean_poss_vector = get_mean_poss_vector(passes_train_outer, k)
    mean_poss_vector_raw = get_mean_poss_vector(passes_train_outer, k, status = False)
    
    index_series = pd.Series(np.zeros(k), index = range(k))
        
    X_raw, X, y = prepare_data(passes_train_outer, mean_poss_vector,
                                      mean_poss_vector_raw, index_series, k)
    
    if model_type == 'status':
        X_model = X
    if model_type == 'raw':
        X_model = X_raw
        
    if regularization == 'full':
        model = LinearRegression()
    else:
        model = ElasticNet(alpha = regularization/50)
    
    #fit model
    model.fit(X_model, y)
    
    #prepare test data
    X_raw_test, X_test, y_test = prepare_data(passes_test, mean_poss_vector,
                              mean_poss_vector_raw, index_series, k)
    if model_type == 'status':
        X_model_test = X_test
    if model_type == 'raw':
        X_model_test = X_raw_test
        
    #score model
    results = model.score(X_model_test, y_test)
    
    if best_model[0] == 'Voronoi':
        return(model, results, k_means)
    else:
        return(model,results)
        

def save_results():
    with open('scores.pickle', 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('scores_rect.pickle', 'wb') as handle:
        pickle.dump(scores_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #save average scores per inner fold
    with open('scores_averages.pickle', 'wb') as handle:
        pickle.dump(scores_averages, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('scores_averages_rect.pickle', 'wb') as handle:
        pickle.dump(scores_averages_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #save all best modles 
    with open('models.pickle', 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('models_rect.pickle', 'wb') as handle:
        pickle.dump(models_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('best_models.pickle', 'wb') as handle:
        pickle.dump(best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('best_models_rect.pickle', 'wb') as handle:
        pickle.dump(best_models_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('models_baseline_raw.pickle', 'wb') as handle:
        pickle.dump(models_baseline_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('models_baseline_status.pickle', 'wb') as handle:
        pickle.dump(models_baseline_status, handle, protocol=pickle.HIGHEST_PROTOCOL)      
    
       
   #save all params best models
    with open('best_params.pickle', 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('best_params_rect.pickle', 'wb') as handle:
        pickle.dump(best_params_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('params_baseline_status.pickle', 'wb') as handle:
        pickle.dump(params_baseline_status, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('params_baseline_raw.pickle', 'wb') as handle:
        pickle.dump(params_baseline_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
   #save results best models
    with open('results_best_models.pickle', 'wb') as handle:
        pickle.dump(results_best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('kmeans_best_models.pickle', 'wb') as handle:
        pickle.dump(kmeans_best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results_best_models_rect.pickle', 'wb') as handle:
        pickle.dump(results_best_models_rect, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results_baseline_raw.pickle', 'wb') as handle:
        pickle.dump(results_baseline_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results_baseline_status.pickle', 'wb') as handle:
        pickle.dump(results_baseline_status, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def calc_mean_r_squared_outer(best_results):
    mean = 0
    for score in best_results.values():
        mean += score
        
    mean = mean/len(best_results.values())
    print(np.round(mean, 3))

def fit_and_score_even(passes_train_outer, passes_test, best_model):
    
    k = best_model[1]
    model_type = best_model[2]
    regularization = best_model[3]
    
    
    if best_model[0] == 'Voronoi':
        passes_train_outer, passes_test, k_means = assign_vor_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    if best_model[0] == 'Rectangular':
        
        passes_train_outer, passes_test = assign_rect_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    if best_model[0] == 'Baseline':
        passes_train_outer, passes_test = assign_rect_zones(passes_train_outer,
                                                                  passes_test,
                                                                  k)
        
    
    mean_poss_vector = get_mean_poss_vector(passes_train_outer, k)
    mean_poss_vector_raw = get_mean_poss_vector(passes_train_outer, k, status = False)
    
    index_series = pd.Series(np.zeros(k), index = range(k))
        
    X_raw, X, y = prepare_data(passes_train_outer, mean_poss_vector,
                                      mean_poss_vector_raw, index_series, k)
    
    if model_type == 'status':
        X_model = X
    if model_type == 'raw':
        X_model = X_raw
        
    if regularization == 'full':
        model = LinearRegression()
    else:
        model = ElasticNet(alpha = regularization/50)
    
    #fit model
    model.fit(X_model, y)
    
    #prepare test data
    X_raw_test, X_test, y_test = prepare_data(passes_test, mean_poss_vector,
                              mean_poss_vector_raw, index_series, k)
    if model_type == 'status':
        X_model_test = X_test
    if model_type == 'raw':
        X_model_test = X_raw_test
        
    #score model
    results = model.score(X_model_test, y_test)
    
    if best_model[0] == 'Voronoi':
        return(model, results, k_means)
    else:
        return(model,results)