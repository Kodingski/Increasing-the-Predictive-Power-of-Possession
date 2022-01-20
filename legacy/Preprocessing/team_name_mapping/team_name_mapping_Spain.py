# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:50:16 2021

@author: timsc
"""

#%%imports
import numpy as np
import pandas as pd


from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#%%load matches spain and odds spain
country = 'Spain'
odds = pd.read_csv(f'data/odds/odds_{country}.csv')
matches = pd.read_json(f"data\matches\matches_{country}.json", encoding='unicode_escape')
teams = pd.read_json(r"data\teams\teams.json", encoding='unicode_escape')

#%%function to extract home and away team from match dict and make it its own column
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
matches['home'] = matches.apply(add_home_away, axis = 1)
matches['away'] = matches.apply(add_home_away, axis = 1, away = True)

#%%
teams = teams[['wyId', 'name', 'officialName']]

team_list =  pd.merge(teams, matches, how = 'inner', 
                      left_on = 'wyId', right_on = 'home' )

team_list_events = team_list.drop_duplicates(subset = 'name')[['name', 'officialName']]

#%% create list of team names used in odds
team_list_odds = odds[['HomeTeam']].drop_duplicates()


#%%
def fuzzy_merge(df_1, df_2, key1, key2, threshold=0, limit=1):
    """
    :param df_1: the left table to join
    :param df_2: the right table to join
    :param key1: key column of the left table
    :param key2: key column of the right table
    :param threshold: how close the matches should be to return a match, based on Levenshtein distance
    :param limit: the amount of matches that will get returned, these are sorted high to low
    :return: dataframe with boths keys and matches
    """
    s = df_2[key2].tolist()
    
    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
    df_1['matches'] = m
    
    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2
    
    return df_1


fuzzy_merge(team_list_events, team_list_odds, 'officialName', 'HomeTeam')

#%%manually correct wrong matches and convert to dict

##one mistake found (atletico wrongly assigned real)

pd.Series(team_list_events.name.values,index=team_list_events.matches).to_dict()

name_mapping = pd.Series(team_list_events.matches.values,index=team_list_events.name).to_dict()

name_mapping['Atlético Madrid'] = 'Ath Madrid'
