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
from utilities.model import * 

#%%load final model
final_model = load_final_model()

#%%Figure 7.2 - plot final model on pitch
plot_weights(final_model['model'], final_model['kmeans'], 11, model_name = f'Final Status')

