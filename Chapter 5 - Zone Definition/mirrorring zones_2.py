# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:23:14 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
import matplotlib.patches as patches
import random


from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *
#For readibility purposes functions to produce zones and plots can be found in
#utils.zones and are imported here
from utilities.zones import *

#%% load passes
passes = load_passes()
    
#%%prepare data 
#use helper to mirror passes
passes_mirrored = mirror_passes(passes)
#reduce upper half of pitch after mirroring
passes_mirrored = passes_mirrored[passes_mirrored['start_y'] <= 50]

#%%Figure 5.1 Mirroring along horizontal axis



#%%Figure 5.2 Mirroring visualized Voronoi
#loop over different values of k
for k in [5,10,15,20]:
    #fit kmeans on passes and plot on pitch
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_mirrored[['start_x', 'start_y']])
    plot_weights(kmeans, k, mirror = True)


#%%Figure 5.3 Mirroring visualized Rectangular
#loop over different values of k
for k in [2,6,8,15,18]:
    #plot on pitch
    plot_rect_zones_mirrored(k)


