# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:46:32 2021

@author: timsc
"""

#%%imports
import numpy as np
import pandas as pd

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
#%%read odds data
country = 'Spain' 
odds = pd.read_csv(f'data/odds/odds_{country}.csv')
matches = pd.read_json(f"data\matches\matches_{country}.json", encoding='unicode_escape')
teams = pd.read_json(r"data\teams\teams.json", encoding='unicode_escape')


#%%reduce data to relevant info
odds = odds[['Div', 'Date', 'HomeTeam',
             'AwayTeam', 'FTHG', 'FTAG', 
             'BbAvH', 'BbAvD', 'BbAvA' ]]

#%%compute reciprocal of sum of avg odds to have way to correct for rake
odds['rake'] = ((1/odds['BbAvH']) +(1/odds['BbAvD']) + (1/odds['BbAvA']))

#%%compute implied winning percentages
odds['percHome'] = (1/odds['BbAvH'])/odds['rake']
odds['percDraw'] = (1/odds['BbAvD'])/odds['rake']
odds['percAway'] = (1/odds['BbAvA'])/odds['rake']

#%% test if correction worked and percentages sum up to 1
test = odds['percHome'] + odds['percDraw'] + odds['percAway']
#works!
##reduce
odds = odds[['Date', 'HomeTeam',
             'AwayTeam', 'FTHG', 'FTAG', 
             'percHome', 'percDraw', 'percAway' ]]

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
teams = teams[['wyId', 'name', 'officialName', 'city']]

team_list =  pd.merge(teams, matches, how = 'inner', 
                      left_on = 'wyId', right_on = 'home' )

team_list_events = team_list.drop_duplicates(subset = 'name')[['name', 'officialName', 'city']]
#%% create list of team names used in odds
team_list_odds = odds[['HomeTeam']].drop_duplicates()
#%% create wyID to name dict
id_name_mapping = pd.Series(teams.name.values,index=teams.wyId).to_dict()


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

name_mapping = pd.Series(team_list_events.matches.values,index=team_list_events.name).to_dict()
#name_mapping['Atlético Madrid'] = 'Ath Madrid'

reversed_name_mapping = {value : key for (key, value) in name_mapping.items()}


#%%apply mapping to odds df
odds['HomeTeam'] = odds['HomeTeam'].map(reversed_name_mapping)
odds['AwayTeam'] = odds['AwayTeam'].map(reversed_name_mapping)


#%%merge team names to match data


matches['home_name'] = matches['home'].map(id_name_mapping)
matches['away_name'] = matches['away'].map(id_name_mapping)

matches = matches.merge(odds, how = 'left', left_on = ('home_name', 'away_name'),
                        right_on = ('HomeTeam', 'AwayTeam'))

##check if dates and results match uo
samp = matches.sample(100)
##looks good

#%%drop unnecessary columns
matches = matches.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], axis = 1)

#%%draw hist of winning percentages to decide on categories
matches['edgeHomeTeam'] = matches['percHome'] - matches['percAway']
#matches['edgeAwayTeam'] = matches['percAway'] - matches['percHome']

#%%
matches['edgeHomeTeam'].hist()

#%%
matches['edgeAwayTeam'].hist()


#%%add labels
def add_matchup_balance(row):
    home_edge = row['edgeHomeTeam']
    
    if home_edge < -0.1:
        return 'away_favored'        
    
    if -0.1 < home_edge < 0.1:
        return 'even'
    if home_edge > 0.1:
        return 'home_favored'


matches['muBalance'] = matches.apply(add_matchup_balance, axis = 1)

#%%save with joined odds
matches.to_json(f"data\matches\matches_{country}_odds.json")
