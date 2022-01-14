# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:28:56 2021

@author: timsc
"""

#%% imports
import numpy as np

from sklearn.cluster import KMeans
from utilities.utils import *
from utilities.model import *

#%% load passes
passes = load_passes()

#%%mirror passes and reduce to bottom half
passes_mirrored = mirror_passes(passes)
passes_mirrored = passes_mirrored[passes_mirrored['start_y'] <= 50]

#%% fit k means to plot zones later
kmeans_10 = KMeans(n_clusters = 10, max_iter = 500).fit(passes_mirrored[['start_x', 'start_y']])

#%%Reduce to sample match
#ID of Match Eintracht Bremen: 2516834
passes_match_example = passes_mirrored[passes_mirrored['matchId'] == 2516834]
passes_match_example['Team'] = 'Home'
passes_match_example.loc[passes_match_example['teamId'] == 2443, 'Team'] = 'Away'

#%%Assign zones to passes
passes_mirrored['zone'] = kmeans_10.predict(passes_mirrored[['start_x','start_y']])

#%%Figure 6.1 Compostion Model
#get mean poss vector to control for mean possession per zone
mean_poss_vector = get_mean_poss_vector(passes_mirrored, 10, status = False)

#create random sample
example_poss_per_zone = mean_poss_vector + np.random.normal(0,0.15, size = 10)

#plot 3 components
plot_poss(kmeans_10, 10, example_poss_per_zone)
plot_poss(kmeans_10, 10, mean_poss_vector)
plot_poss(kmeans_10, 10, example_poss_per_zone - mean_poss_vector, colors = True)

#%%Figure 6.2 Composition Model Status Controlled
#get mean poss vector to control for mean possession per zone
mean_poss_vector_status = get_mean_poss_vector(passes_mirrored, 10, status = True)

#create random sample
example_poss_per_zone_per_status = mean_poss_vector_status + np.random.normal(0,0.15, size = 30)

#plot 3x3 components
plot_poss(kmeans_10, 10, example_poss_per_zone_per_status, status = True)
plot_poss(kmeans_10, 10, mean_poss_vector_status, status = True)
plot_poss(kmeans_10, 10, example_poss_per_zone_per_status - mean_poss_vector_status, 
          status = True, colors = True)