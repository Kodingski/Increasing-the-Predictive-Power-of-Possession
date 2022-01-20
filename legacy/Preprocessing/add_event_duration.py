# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:23:20 2021

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
events_spain = pd.read_json('events_spain_score_outcome_pos_success.json')

#%%
def add_event_duration(events):
    
    events['eventDuration'] = False
    def add_event_duration_match(match_events):
    
        for ht in ['1H','2H']:    
            match_events.loc[match_events['matchPeriod'] == ht,'eventDuration'] = (
                                   -match_events.loc[match_events['matchPeriod'] == ht, 'eventSec'].diff(-1).fillna(0) 
                                   )
        return(match_events)    
    
    
    for _, match_events in events.groupby(['matchId']):
        events.iloc[match_events.index] = add_event_duration_match(match_events)
    return(events)

#%%
events_spain = add_event_duration(events_spain)

#%%
events_spain.to_json('events_spain_score_outcome_pos_success_dur.json')
