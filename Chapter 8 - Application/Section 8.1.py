# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:19:26 2021

@author: timsc
"""
#%% imports
import scipy.stats
import scipy.optimize
import seaborn as sns
import math

from utilities.application import *
from utilities.utils import *
from utilities.plot_utils import *

#%%load match overview
matches_overview = load_postprocessed()
matches_overview = matches_overview.drop('matchId', axis = 1)

#%%helper to calculate f statistic of Kolmogorov-Smirnov
def calc_f(x):
    f = scipy.stats.ks_2samp(matches_overview['raw poss. Home'], 
                   matches_overview['pred Home'].apply(lambda data : (1 / (1 + math.exp(-x*data)))))[0]

    return(f)

#%%optimize f statistic for different values of scaling factor to find optimal factor
optimization = scipy.optimize.minimize_scalar(calc_f)
minimal_x = optimization['x']

#%%add transformed pred to df
matches_overview['pred Home trans'] = matches_overview['pred Home'].apply(lambda x : (1 / (1 + math.exp(-minimal_x*x))))

#%%Figure 8.1 - CDF of transformed vs raw possession
#prepare data to plot with sns
data_pred = matches_overview[['pred Home trans']]
data_pred['hue'] = 'Transformed Model Pred.'
data_pred.columns = ['Possession', 'Type']

data_raw = matches_overview[['raw poss. Home']]
data_raw['hue'] = 'Raw Possession'
data_raw.columns = ['Possession', 'Type']

data_plot = data_pred.append(data_raw, ignore_index = True)

#plot
sns.set_theme()
ax = sns.ecdfplot(data = data_plot, x = 'Possession', hue = 'Type')
ax.set(ylim=(-0.05, 1.05))



