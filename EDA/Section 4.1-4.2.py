# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:00:05 2021

@author: timsc
"""
#%%imports
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *


#%% load data and reduce to passes played in even matches
passes = load_passes()
passes_even = passes.loc[passes['muBalance'] == 'even']

#%%function to produce results of these sections
def event_per_outcome(data, event = 'Pass', n_zones = False):
    event_data = data[data['eventName'] == event]
    event_grouped = event_data.groupby(['outcome'])
    outcome_dict = {}
    
    

    for name, group in event_grouped:
        
        if not n_zones:
            outcome_dict[name] = len(group)
            
        else:    
            zones = range(n_zones)
            sum_events = len(group)
            outcome_dict[name] = []
        
            for thresh in zones:
                
                events_in_zone = len(group[((group['start_x'] >= (thresh/n_zones)*100) & (group['start_x'] <= (thresh+1)/n_zones*100))])
                outcome_dict[name].append(round((events_in_zone/sum_events), 2))
            
            outcome_dict[name].append(sum_events)
    
    return(outcome_dict)
    

##calculate number of passes per status
def event_per_status(data, event = 'Pass'):
    
    event_data = data[data['eventName'] == event]
    event_grouped = event_data.groupby(['status'])

    status_dict = {}
        
    for name, group in event_grouped:
             
        status_dict[name] = len(group)
    
    return(status_dict)



##calculate number of passes per status per outcome
def event_per_outcome_status(data, event = 'Pass'):
    
    event_data = data[data['eventName'] == event]
    event_grouped = event_data.groupby(['status', 'outcome'])

    status_dict = {}
        
    for name, group in event_grouped:
        status = name[0]
        outcome = name[1]
        
        if status not in status_dict.keys():
            status_dict[status] = {}
    
     
        status_dict[status][outcome] = len(group)
    
    return(status_dict)





#%%Table 4.1 (passes per outcome)
passes_per_outcome = event_per_outcome(passes)
average_possession_outcome = (passes_per_outcome['lost']/ 
                            (passes_per_outcome['lost'] + passes_per_outcome['won'])
                            )


#%%Table 4.2 (passes per status)
passes_status = event_per_status(passes)
average_possession_status = (passes_status['trailing']/ 
                            (passes_status['trailing'] + passes_status['leading'])
                            )

#%%Table 4.3 (passes per outcome status = drawing)
passes_per_outcome_status = event_per_outcome_status(passes)
average_possession_status_outcome = (passes_per_outcome_status['drawing']['lost']/ 
                            (passes_per_outcome_status['drawing']['lost'] + passes_per_outcome_status['drawing']['won'])
                            )



#%% Table 4.4 (passes per outcome MU = even)
passes_per_outcome_mu = event_per_outcome(passes_even)
average_possession_outcome_mu = (passes_per_outcome_mu['lost']/ 
                            (passes_per_outcome_mu['lost'] + passes_per_outcome_mu['won'])
                            )

#%% Table 4.5 (passes per outcome MU = even, Status = Drawing)
passes_per_outcome_mu_status = event_per_outcome_status(passes_even)
average_possession_outcome_mu_status = (passes_per_outcome_mu_status['drawing']['lost']/
                            (passes_per_outcome_mu_status['drawing']['lost'] + passes_per_outcome_mu_status['drawing']['won'])
                            )




