# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:28:42 2021

@author: timsc
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

#%%
events_spain = pd.read_json(r"D:\Drive\Leiden\4. Semester\Thesis\event data\events\events_Spain.json")
matches_spain = pd.read_json(r"D:\Drive\Leiden\4. Semester\Thesis\event data\matches\matches_Spain.json")

#%%
events_spain_sample = events_spain.sample(100000)
events_spain_sample['own_score'] = 0
events_spain_sample['opp_score'] = 0

#%%
events_2565922 = events_spain.loc[events_spain['matchId'] == 2565922,]
events_2565922['own_score'] = 0
events_2565922['opp_score'] = 0
#%% function to add current score to every event
def add_current_score(row, events):
    goal = False
    match_id = row.matchId
    #print(row.eventSec)
    time = row.eventSec
    #print(time)
    tags_list = row.tags    
    team_id = row.teamId
    event_id = row.eventId
    ht = row.matchPeriod
    
    for tag in tags_list:
        #print(tag)
        if tag['id'] == 101 and (event_id == 10):
            goal = True
            print('goal found')
    
    if ((goal == True) and (ht == '1H')):
        #add goal to all events that happened later in the first half, plus
        #all events that happend in the second half
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time)
                    & (events['matchPeriod'] == '1H')
                    & (events['teamId'] == team_id)), 'own_score'] += 1
        events.loc[((events['matchId'] == match_id) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] == team_id)), 'own_score'] += 1 
        
        #do the same for opposing team events
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time)
                    & (events['matchPeriod'] == '1H')
                    & (events['teamId'] != team_id)), 'opp_score'] += 1
        events.loc[((events['matchId'] == match_id) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] != team_id)), 'opp_score'] += 1         
        
    if ((goal == True) and (ht == '2H')):
        print('here')
        #add goal to all events that happened later in the second half
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] == team_id)), 'own_score'] += 1

        #do the same for opposing team events
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] != team_id)), 'opp_score'] += 1
    
#%%
events_spain['own_score'] = 0
events_spain['opp_score'] = 0


#%%apply function to whole dataset
events_spain.apply(add_current_score, axis = 1, events = events_spain)

#%%add game status to every event
def add_current_status(row):
    if row.own_score > row.opp_score:
        return 'leading'
    if row.own_score == row.opp_score:
        return 'drawing'
    if row.own_score < row.opp_score:
        return 'trailing'    
#%%apply function to whole dataset
events_spain['status'] = events_spain.apply(add_current_status, axis = 1)


#%%
events_2565569 = events_spain.loc[events_spain['matchId'] == 2565569,]


#investiga different sample match to see if everything worked as intended
#%%
events_spain.to_json('events_spain_score.json')
    
