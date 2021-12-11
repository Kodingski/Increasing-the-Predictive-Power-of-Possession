# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:11:16 2021

@author: timsc
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from utils import *
from plot_utils import *
from metrics import *

#%%
events_spain = pd.read_json('events_spain_score_outcome_pos.json')

#%%
def add_event_success(row):
    event_tags = row.tags
    event_type = row.eventName
    
    for tag in event_tags:
        
        if event_type == 'Duel':
            if 701 in tag.values():
                return 'Successful'
            if 702 in tag.values():
                return 'noWinner'
            if 703 in tag.values():
                return 'Unsuccessful'        
        
        if 1801 in tag.values():
            return 'Successful'
        if 1802 in tag.values():
            return 'Unsuccessful'
    
    return 'noWinner'

#%%
events_spain['eventSuccess'] = events_spain.apply(add_event_success, 
                                                          axis = 1)
#%%
events_spain.to_json('events_spain_score_outcome_pos_success.json')
