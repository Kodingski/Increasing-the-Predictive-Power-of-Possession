
import pandas as pd
import pickle
import os




#path in which you store the downloade data preprocessed for eda and fitting
path_preprocessed = 'data/preprocessed/'
#path in which you store the downloaded model fits
path_results = 'data/models/'
#path in which you store the downloaded data postprocessed for application
path_postprocessed ='data/postprocessed'


def load_events(path_preprocessed = path_preprocessed, countries = ['Germany', 'Spain',
                                   'Italy', 'England', 'France']):    
    #load data, merge and reduce to passes
    events = []
    
    for country in countries:
        events_country = pd.read_json(rf'{path}events_{country}_thesis.json')
        events_country['country'] = country
        events.append(events_country)
    
    del events_country
    
    events = pd.concat(events, axis=0, ignore_index=True)

    # reduce to passes for quicker computation
    return(events)




def load_passes(path_preprocessed = path_preprocessed, countries = ['Germany', 'Spain',
                                   'Italy', 'England', 'France']):    
    #load data, merge and reduce to passes
    events = []
    
    for country in countries:
        events_country = pd.read_json(rf'{path_preprocessed}events_{country}_thesis.json')
        events_country['country'] = country
        events.append(events_country)
    
    del events_country
    
    events = pd.concat(events, axis=0, ignore_index=True)

    # reduce to passes for quicker computation
    passes = events[events['eventName'] == 'Pass']
    return(passes)

def load_matches(path_preprocessed = path_preprocessed):
    with open(rf'{path_preprocessed}matches_preprocessed.json', 'rb') as f:
       matches = pd.read_json(f)
    return matches

def load_final_model(path_results = path_results):
    model_parts = ['model', 'kmeans', 'mean_poss_vector']
    final_model = dict.fromkeys(model_parts)
    for part in model_parts:
        with open(rf'{path_results}final/{part}.pickle', 'rb') as f:
            final_model[part] = pickle.load(f)

    return final_model


def load_full_results(path_results = path_results):
    files = os.listdir(rf'{path_results}full')
    results = {}
    
    
    for f in files:
        if f == 'models.pickle':
            models = pickle.load(open(rf'{path_results}full/{f}', "rb"))
        else:                        
            results[str(f.rstrip('.pickle'))] = pickle.load(open(rf'{path_results}full/{f}', "rb"))
            
    return(models, results)

def load_test_even(path_results = path_results):
    with open(rf'{path_results}even/test_even.pickle', 'rb') as f:
       test_even = pickle.load(f)
    return test_even