# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:47:51 2022

@author: timsc
"""

#%%
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
from Utilities.utils import *
from Utilities.plot_utils import *
from Utilities.metrics import *

#%% load data, merge and reduce to passes
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)

passes = events[events['eventName'] == 'Pass']