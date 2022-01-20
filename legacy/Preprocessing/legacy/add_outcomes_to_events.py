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
events_spain = pd.read_json(r"events_spain_score.json")
matches_spain = pd.read_json(r"matches_Spain.json")

#%%make every positionial info into its own column
events_lists = pd.DataFrame(events_spain.positions.tolist(), index= events_spain.index)
events_lists.iloc[:,1] = events_lists.iloc[:,1].apply(lambda x: {} if pd.isna(x) else x)

start = pd.json_normalize(events_lists.iloc[:,0])
end  = pd.json_normalize(events_lists.iloc[:,1])

events_spain[['start_y', 'start_x']] = start
events_spain[['end_y', 'end_x']] = end

#%%
teams_data = matches_spain.teamsData

def add_home_away(row, away = False):
    teams_data = row.teamsData
    for team, team_dict in teams_data.items():
        if team_dict['side'] == 'home':
            home_team = int(team)
        if team_dict['side'] == 'away':
            away_team = int(team)
          
    if away:
        return(away_team)
    
    return(home_team)

#%%
matches_spain['home'] = matches_spain.apply(add_home_away, axis = 1)
matches_spain['away'] = matches_spain.apply(add_home_away, axis = 1, away = True)

#%%reduce matches to match_id and winner and join it to events
matches_spain = matches_spain[['wyId', 'home', 'away', 'winner']] 
events_spain = events_spain.merge(matches_spain, how = 'inner', 
                              left_on = 'matchId', right_on = 'wyId'
                              )

#%%
events_spain_sample = events_spain.sample(1000)
#%%
events_spain.to_json('events_spain_score_outcome_pos.json')


