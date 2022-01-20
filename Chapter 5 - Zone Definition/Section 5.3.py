# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 13:07:02 2021

@author: timsc
"""

#%% imports
from sklearn.cluster import KMeans
from utilities.utils import *
from utilities.plot_utils import *
from utilities.zones import *

#%% load data and reduce to passes of even matches
passes = load_passes()
passes_even = passes.loc[passes['muBalance'] == 'even']

#%% mirror and reduce to passes on bottom half
passes_mirrored = mirror_passes(passes_even)
passes_mirrored = passes_mirrored[passes_mirrored['start_y'] <= 50]

#%%Figure 5.4A - mean poss per zone Vor
#fit kmeans for k = 10
k = 10
kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_mirrored[['start_x', 'start_y']])

#calculate mean poss per zone based on k_means
mean_poss_per_zone = get_mean_poss_per_zone(passes_mirrored, k, kmeans)

#plot on pitch
plot_mean_poss_per_zone(mean_poss_per_zone, kmeans, k)

#%%Figure 5.4B - mean poss per zone Rect
#calculate mean poss per zone vor k = 8
k_rect = 8
mean_poss_vector_rect = get_mean_poss_per_zone(passes_even, 8)

#plot on pitch
plot_mean_poss_per_rect_zone(mean_poss_vector_rect, k_rect)

