# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:41:51 2021

@author: timsc
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
#%%load events and matches for one league to preprocess
country = 'France' ##Set this string to country of league to preprocess
events = pd.read_json(f"data\events\events_{country}.json")
matches = pd.read_json(f"data\matches\matches_{country}_odds.json")

#%% function to add current score to every event
def add_current_score(row, events):

    goal = False
    own_goal = False
    match_id = row.matchId
    time = row.eventSec
    tags_list = row.tags    
    team_id = row.teamId
    event_id = row.eventId
    ht = row.matchPeriod
    
    for tag in tags_list:
        #print(tag)
        if tag['id'] == 101 and (event_id == 10 or event_id == 3):
            goal = True
        if tag['id'] == 102:
            own_goal = True
    
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
        
    if ((own_goal == True) and (ht == '1H')):
        #add goal to all events that happened later in the first half, plus
        #all events that happend in the second half
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time)
                    & (events['matchPeriod'] == '1H')
                    & (events['teamId'] == team_id)), 'opp_score'] += 1
        events.loc[((events['matchId'] == match_id) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] == team_id)), 'opp_score'] += 1 
        
        #do the same for opposing team events
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time)
                    & (events['matchPeriod'] == '1H')
                    & (events['teamId'] != team_id)), 'own_score'] += 1
        events.loc[((events['matchId'] == match_id) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] != team_id)), 'own_score'] += 1         
        
    if ((own_goal == True) and (ht == '2H')):
        #add goal to all events that happened later in the second half
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] == team_id)), 'opp_score'] += 1

        #do the same for opposing team events
        events.loc[((events['matchId'] == match_id) 
                    & (events['eventSec'] > time) 
                    & (events['matchPeriod'] == '2H')
                    & (events['teamId'] != team_id)), 'own_score'] += 1
        
#%%add scores columns to events df and fill them via applying function
events['own_score'] = 0
events['opp_score'] = 0
events.apply(add_current_score, axis = 1, events = events)

#%%function to create categorical match status       
def add_current_status(row):
    if row.own_score > row.opp_score:
        return 'leading'
    if row.own_score == row.opp_score:
        return 'drawing'
    if row.own_score < row.opp_score:
        return 'trailing'
#%%apply function to whole dataset
events['status'] = events.apply(add_current_status, axis = 1)
#%%function to determine success of each event
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
#%%apply function to whole dataset
events['eventSuccess'] = events.apply(add_event_success, axis = 1)
#%%make every positionial info into its own column
def add_positions(events):

    events_lists = pd.DataFrame(events.positions.tolist(), index= events.index)
    events_lists.iloc[:,1] = events_lists.iloc[:,1].apply(lambda x: {} if pd.isna(x) else x)
    
    start = pd.json_normalize(events_lists.iloc[:,0])
    end  = pd.json_normalize(events_lists.iloc[:,1])
    
    events[['start_y', 'start_x']] = start
    events[['end_y', 'end_x']] = end
    
    return(events)     
#%%apply function to whole dataset
events = add_positions(events)

matches = matches[['wyId', 'home', 'away', 'winner', 
                   'percHome', 'percDraw', 'percAway', 'muBalance']] 
events = events.merge(matches, how = 'inner', 
                              left_on = 'matchId', right_on = 'wyId'
                              ).drop('wyId', 1)

#%%add column indicating outcome 
def add_match_outcome(row):
    if row.winner == 0:
        return 'drew' 
    if row.teamId == row.winner:
        return 'won'
    if row.teamId != row.winner:
        return 'lost'
 
#%%apply function to whole dataset
events['outcome'] = events.apply(add_match_outcome, axis = 1)
#%%function to add duration of each event to event dataframe
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

#%%apply function to whole dataset
events = add_event_duration(events)

#%%save preprocess file
events.to_json(f"data\preprocessed\events_{country}_preprocessed.json")


