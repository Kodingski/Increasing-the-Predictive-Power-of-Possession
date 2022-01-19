# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:27:53 2021

@author: timsc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:47:36 2021

@author: timsc
"""
#%% imports
import numpy as np
import matplotlib as plt
import matplotlib.patches as patches
import pickle

from utilities.utils import *
from utilities.plot_utils import *

#%%
final_model = load_final_model()

#%%
plot_weights(final_model['model'], final_model['kmeans'], 11, model_name = f'Final Status')

#%%s   
with open('final_kmeans.pickle', 'wb') as handle:
    pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('final_model.pickle', 'wb') as handle:
    pickle.dump(final_model_s, handle, protocol=pickle.HIGHEST_PROTOCOL) 
