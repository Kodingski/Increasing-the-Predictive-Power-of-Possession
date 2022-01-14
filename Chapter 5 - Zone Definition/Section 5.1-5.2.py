# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:23:14 2021

@author: timsc
"""

#%% imports

import seaborn as sns
import matplotlib as plt



from utilities.utils import *
from utilities.plot_utils import *
#For readibility purposes functions to produce zones and plots can be found in
#utilities.zones and are imported here
from utilities.zones import *

#%% load passes
passes = load_passes()
    
#%%prepare data 
#use helper to mirror passes
passes_mirrored = mirror_passes(passes)
#reduce upper half of pitch after mirroring
passes_mirrored = passes_mirrored[passes_mirrored['start_y'] <= 50]

#%%Figure 5.1 Mirroring along horizontal axis visualized
#draw sample of n = 1000
samp = passes.sample(1000)

#plot sample on pitch
fig, ax = pitch()
sns.scatterplot('start_x', 'start_y', data=samp)
plt.xlim(-1,101)
plt.ylim(-1,101)
plt.axis('off')
fig.tight_layout()
plt.show()

#mirror sample
samp_mirrored = mirror_passes(samp)
samp_mirrored = samp_mirrored[samp_mirrored['start_y'] <= 50]

#plot mirrored on pitch
fig, ax = pitch()
sns.scatterplot('start_x', 'start_y', data=samp_mirrored)
plt.axhline(50, color ='black')
plt.xlim(-1,101)
plt.ylim(-1,101)
plt.axis('off')
fig.tight_layout()
plt.show()

#plot vor zones for k = 10, unmirrored
plot_weights(kmeans, 10, mirror = False)
         
#plot vor zones for k = 10, mirrored
plot_weights(kmeans, 10, mirror = True)

#%%Figure 5.2 Mirroring visualized Rectangular
#loop over different values of k
for k in [2,6,8,15,18]:
    #plot on pitch
    plot_rect_zones_mirrored(k)

#%%Figure 5.3 Mirroring visualized Voronoi
#loop over different values of k
for k in [5,10,15,20]:
    #fit kmeans on passes and plot on pitch
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_mirrored[['start_x', 'start_y']])
    plot_weights(kmeans, k, mirror = True)

