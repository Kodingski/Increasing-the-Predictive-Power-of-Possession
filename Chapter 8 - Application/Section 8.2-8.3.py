# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:19:26 2021

@author: timsc
"""
#%% imports
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import math

from utilities.application import *
from utilities.utils import *
from utilities.plot_utils import *

#%%load match overview
matches_overview = load_postprocessed()
matches_overview = matches_overview.drop('matchId', axis = 1)

#%%load matches
matches = load_matches()

#%%load final model
final_model = load_final_model()

#%%helper to calculate f statistic of Kolmogorov-Smirnov
def calc_f(x):
    f = scipy.stats.ks_2samp(matches_overview['raw poss. Home'], 
                   matches_overview['pred Home'].apply(lambda data : (1 / (1 + math.exp(-x*data)))))[0]

    return(f)

#%%optimize f statistic for different values of scaling factor to find optimal factor
optimization = scipy.optimize.minimize_scalar(calc_f)
minimal_x = optimization['x']

#%%add transformed pred to df and calculate diff
matches_overview['pred Home trans'] = matches_overview['pred Home'].apply(lambda x : (1 / (1 + math.exp(-minimal_x*x))))

#calculate absolute difference between predictions
matches_overview['diff'] = np.abs(matches_overview['pred Home trans'] - matches_overview['raw poss. Home'])
            
#%%reduce matches to la liga
matches_la_liga = matches[matches['roundId'] == 4406122] 
#%%Table 8.1 - Table La Liga
matches_la_liga_overview = pd.merge(matches_la_liga[['wyId', 'home_name', 'away_name']], matches_overview, how = 'left', left_on = 'wyId', right_on = 'wyId')

#get poss per team
home_poss_per_team = matches_la_liga_overview.groupby('home_name').mean()
away_poss_per_team = matches_la_liga_overview.groupby('away_name').mean()
away_poss_per_team.iloc[:,0:4] = 1 - away_poss_per_team.iloc[:,0:4]
away_poss_per_team.iloc[:,4:38] = away_poss_per_team.iloc[:,4:38]*-1
away_poss_per_team.iloc[:,38] = 1 - away_poss_per_team.iloc[:,38]

poss_per_team = ((home_poss_per_team + away_poss_per_team) / 2).reset_index()
poss_per_team['diff'] = poss_per_team['raw poss. Home'] - poss_per_team['pred Home trans']

#build table la liga
table_la_liga = poss_per_team[['home_name', 'raw poss. Home', 'pred Home trans', 'diff']]
table_la_liga.columns = ['Club Name', 'Raw Poss.', 'Trans. Model Pred.', 'Difference']

table_la_liga['Position'] = [16,2,1,13,14,18,9,11,8,10,19,17,15,20,6,3,12,7,4,5]
table_la_liga['Points'] =[43,79,93,49,47,29,51,49,55,51,22,43,46,20,60,76,49,58,73,61]
table_la_liga['Goal Diff.'] = [-8,36,70,-1,-10,-38,-6,-6,9,-9,-50,-17,-14,-37,-1,50,7,-9,27,7]

col_order = ['Position', 'Club Name','Goal Diff.', 'Points', 'Raw Poss.', 'Trans. Model Pred.', ]

#latex table
table_latex_ready = table_la_liga[col_order]

#%%Figure 8.2-8.3 - poss profile Atletico
poss_atletico = poss_per_team[poss_per_team['home_name'] == 'Atlético Madrid']
X_atletico = poss_atletico.iloc[0,6:39].values

plot_weights_post(final_model['model'], final_model['kmeans'], X_atletico, 
                  coeffs_inc = False)
plot_weights_post(final_model['model'], final_model['kmeans'], X_atletico)


#%%Figure 8.4-8.5 - poss profile Las Palmas
poss_las_palmas = poss_per_team[poss_per_team['home_name'] == 'Las Palmas']
X_las_palmas = poss_las_palmas.iloc[0,6:39].values

plot_weights_post(final_model['model'], final_model['kmeans'], X_las_palmas,
                  coeffs_inc = False)
plot_weights_post(final_model['model'], final_model['kmeans'], X_las_palmas)


#%% Table 8.2 - Matches with biggest diff
#build latex table matches
matches_latex = pd.merge(matches[['wyId', 'home_name', 'away_name', 'gameweek', 'date']],
                         matches_overview[['wyId', 'label', 'home_score', 
                                           'away_score', 'diff',
                                           'raw poss. Home', 'pred Home trans']],
                         how = 'left', left_on = 'wyId', right_on = 'wyId')

matches_latex = matches_latex.sort_values('diff', ascending = False).head(20)
matches_latex_red = matches_latex[['label', 'date', 'gameweek', 'raw poss. Home', 
                                   'pred Home trans', 'diff']]
matches_latex_red[['Matchup', 'Result']] = matches_latex_red['label'].str.split(',', expand = True)
matches_latex_red['Result'] = matches_latex_red['Result'].str.strip()

matches_latex_red['Date'] = matches_latex_red['date'].str.split('at', expand = True)[0]
matches_latex_red['Date'] = matches_latex_red['Date'].str.strip()

matches_latex = matches_latex_red[['Matchup', 'gameweek', 'Result', 'raw poss. Home', 
                                   'pred Home trans', 'diff']]
matches_latex.columns = ['Match-up', 'Match-day', 'Result', 'Raw Poss.', 'Trans. Model Pred.', 'Difference']

matches_latex_str = matches_latex.to_latex(float_format="%.2f", index = False)

#final table
table_latex = table_latex_ready.sort_values('Position').to_latex(float_format="%.2f", index = False)


#%%Figure 8.7 - 8.9 Analysis poss Betis - Real
#select match and poss
betis_real = matches_overview[matches_overview['wyId'] == 2565780]
X_betis_real = betis_real.iloc[0,6:39].values

plot_weights_post(final_model['model'], final_model['kmeans'], X_betis_real)



